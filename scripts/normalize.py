import os
import re
import json
import gzip
import sqlite3
import hashlib
import logging
import zipfile
import tempfile
from pathlib import Path
from collections import Counter, defaultdict

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "quarantine"
REPORTS = ROOT / "reports"
TRAINING = ROOT / "training"
DB_PATH = ROOT / "dedup.sqlite"

for directory in (OUT, REPORTS, TRAINING):
    directory.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

SOURCES = [
    {"id": "AletheiaResearch/GPT-5.5-Codex", "category": "coding"},
    {"id": "greghavens/gpt-5.6-sol-coding-and-debugging-traces", "category": "agent"},
    {"id": "sollamon/ox-alpha-glm-5.3-flash-distillation-coding-17k-raw", "category": "coding"},
    {"id": "Crownelius/GLM-5.0-25000x", "category": "reasoning"},
    {"id": "saidutta69/fable-5-premium-v2", "category": "reasoning"},
    {"id": "DSFFGFG456/fable-5-coding-and-debugging-traces", "category": "agent"},
    {"id": "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x", "category": "reasoning"},
    {"id": "saidutta69/claude-mythos-distilled-25k-clean", "category": "reasoning"},
    {"id": "faunix/Qwen3.8-27B-Distillation-40K", "category": "general"},
    {"id": "Jackrong/DeepSeek-V4-Distill-8000x", "category": "reasoning"},
]

collection_stats = Counter()
collection_errors = []
source_reports = []


def text_value(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "\n".join(parts)
    if value is None:
        return ""
    try:
        return json.dumps(value, ensure_ascii=False)
    except (TypeError, ValueError):
        return str(value)


def flatten_content(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(flatten_content(x) for x in value)
    if isinstance(value, dict):
        if isinstance(value.get("text"), str):
            return value["text"]
        return json.dumps(value, ensure_ascii=False)
    return str(value) if value is not None else ""


def normalize_messages(record):
    messages = record.get("messages")
    if not isinstance(messages, list):
        messages = record.get("conversations")

    if isinstance(messages, list):
        result = []
        role_map = {
            "human": "user",
            "user": "user",
            "gpt": "assistant",
            "bot": "assistant",
            "model": "assistant",
            "assistant": "assistant",
            "system": "system",
            "developer": "developer",
            "tool": "tool",
            "function": "tool",
        }

        for item in messages:
            if not isinstance(item, dict):
                continue

            raw_role = item.get("role") or item.get("from") or item.get("speaker")
            role = role_map.get(str(raw_role).lower()) if raw_role else None
            if not role:
                continue

            content = item.get(
                "content",
                item.get("value", item.get("text", ""))
            )

            message = {
                "role": role,
                "content": content if isinstance(content, (str, list, dict))
                else text_value(content),
            }

            for key in ("name", "tool_call_id", "tool_calls", "function_call"):
                if key in item:
                    message[key] = item[key]

            result.append(message)

        if result:
            return result

    instruction = text_value(
        record.get("instruction", record.get("prompt", ""))
    )
    extra_input = text_value(record.get("input", ""))
    answer = text_value(
        record.get(
            "output",
            record.get(
                "response",
                record.get("completion", record.get("chosen", ""))
            )
        )
    )

    if instruction and answer:
        user_text = instruction
        if extra_input:
            user_text += "\n\n" + extra_input

        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer},
        ]

    question = text_value(
        record.get("question", record.get("problem", ""))
    )
    answer = text_value(
        record.get("answer", record.get("solution", ""))
    )

    if question and answer:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ]

    return None


def english_check(messages, record=None):
    declared = ""

    if isinstance(record, dict):
        declared = str(
            record.get("language", record.get("lang", ""))
        ).lower().strip()

    allowed = ("en", "eng", "english", "en-us", "en-gb")

    if declared and declared not in allowed:
        return False

    content = " ".join(
        flatten_content(m.get("content", ""))
        for m in messages
    )

    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", content):
        return False

    letters = re.findall(r"[A-Za-z]", content)
    other_letters = re.findall(
        r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]",
        content
    )

    if other_letters:
        ratio = len(other_letters) / max(
            1, len(letters) + len(other_letters)
        )
        if ratio > 0.20:
            return False

    return bool(letters)


def make_record(row, source, index, dedup_db):
    if not isinstance(row, dict):
        return None

    messages = normalize_messages(row)
    if not messages:
        return None

    roles = {m["role"] for m in messages}

    if "user" not in roles or "assistant" not in roles:
        return None

    if not english_check(messages, row):
        collection_stats["non_english"] += 1
        return None

    canonical = json.dumps(
        messages,
        ensure_ascii=False,
        sort_keys=True
    )

    fingerprint = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()

    try:
        dedup_db.execute(
            "INSERT INTO seen (fingerprint) VALUES (?)",
            (fingerprint,)
        )
    except sqlite3.IntegrityError:
        collection_stats["duplicates"] += 1
        return None

    metadata = {
        "source_dataset": source["id"],
        "source_id": str(row.get("id", index)),
        "category": source["category"],
        "language": "en",
        "rights_status": "unverified_quarantine",
    }

    for key in ("teacher_model", "model", "difficulty"):
        if row.get(key) is not None:
            metadata[key] = row[key]

    return {
        "messages": messages,
        "metadata": metadata,
    }


def get_splits(dataset):
    if hasattr(dataset, "keys"):
        return [
            (str(name), split)
            for name, split in dataset.items()
        ]

    return [("train", dataset)]


def collect_source(source, dedup_db):
    repo = source["id"]
    logging.info("Collecting: %s", repo)

    source_stats = Counter()
    safe_name = repo.replace("/", "__")
    output_path = OUT / f"{safe_name}.jsonl.gz"
    token = os.environ.get("HF_TOKEN") or None

    try:
        dataset = load_dataset(
            repo,
            streaming=True,
            token=token
        )

        with gzip.open(
            output_path,
            "wt",
            encoding="utf-8"
        ) as output:

            for split_name, split in get_splits(dataset):
                logging.info("%s: split=%s", repo, split_name)

                for index, row in enumerate(split):
                    source_stats["read"] += 1
                    collection_stats["read"] += 1

                    try:
                        item = make_record(
                            row,
                            source,
                            index,
                            dedup_db
                        )

                        if item is None:
                            source_stats["rejected"] += 1
                            continue

                        output.write(
                            json.dumps(
                                item,
                                ensure_ascii=False
                            ) + "\n"
                        )

                        source_stats["accepted"] += 1
                        collection_stats["accepted"] += 1

                        if collection_stats["read"] % 5000 == 0:
                            dedup_db.commit()
                            logging.info(
                                "Collection progress: %d",
                                collection_stats["read"]
                            )

                    except Exception as exc:
                        source_stats["errors"] += 1
                        collection_errors.append({
                            "dataset": repo,
                            "row": index,
                            "error": str(exc)[:1000],
                        })

        dedup_db.commit()

        result = {
            "dataset": repo,
            "file": str(output_path.relative_to(ROOT)),
            "status": "success",
            "statistics": dict(source_stats),
        }

    except Exception as exc:
        dedup_db.commit()

        result = {
            "dataset": repo,
            "file": str(output_path.relative_to(ROOT))
            if output_path.exists() else None,
            "status": "failed",
            "error": str(exc)[:2000],
            "partial_file_may_exist": output_path.exists(),
            "statistics": dict(source_stats),
        }

        collection_errors.append(result)
        logging.exception("Dataset failed: %s", repo)

    source_reports.append(result)
    logging.info("Finished %s: %s", repo, dict(source_stats))


def inspect_sample(record):
    reasons = []
    messages = record.get("messages")

    if not isinstance(messages, list) or not messages:
        return 0, ["invalid_messages"]

    user_parts = []
    assistant_parts = []

    for message in messages:
        if not isinstance(message, dict):
            reasons.append("invalid_message_object")
            continue

        role = message.get("role")
        content = flatten_content(
            message.get("content", "")
        ).strip()

        if role == "user":
            user_parts.append(content)
        elif role == "assistant":
            assistant_parts.append(content)

    user_text = "\n".join(user_parts).strip()
    assistant_text = "\n".join(assistant_parts).strip()
    full_text = user_text + "\n" + assistant_text

    if not user_text:
        reasons.append("empty_user")

    if not assistant_text:
        reasons.append("empty_assistant")

    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", full_text):
        reasons.append("japanese_script")

    if not re.search(r"[A-Za-z]", full_text):
        reasons.append("no_english_letters")

    if len(full_text) < 80:
        reasons.append("very_short_sample")

    if len(assistant_text) < 25:
        reasons.append("very_short_answer")

    if len(full_text) > 120000:
        reasons.append("extreme_length")

    lowered = assistant_text.lower()

    low_signal_patterns = (
        "as an ai language model",
        "i cannot assist with that request",
        "lorem ipsum",
        "insert your code here",
        "todo: implement",
    )

    if any(p in lowered for p in low_signal_patterns):
        reasons.append("boilerplate_or_placeholder")

    if assistant_text.endswith(
        ("...", "…", "[truncated]", "<|endoftext|>")
    ):
        reasons.append("possible_truncation")

    hard_fail = {
        "empty_user",
        "empty_assistant",
        "invalid_messages",
        "invalid_message_object",
        "japanese_script",
        "no_english_letters",
        "extreme_length",
    }

    if any(reason in hard_fail for reason in reasons):
        return 0, reasons

    penalties = {
        "very_short_sample": 20,
        "very_short_answer": 20,
        "boilerplate_or_placeholder": 35,
        "possible_truncation": 20,
    }

    score = 100

    for reason in reasons:
        score -= penalties.get(reason, 0)

    return max(0, score), reasons


def validate_quarantine():
    logging.info("Starting quality screening")

    stats = Counter()
    per_source = defaultdict(Counter)
    reason_counts = Counter()

    training_path = TRAINING / "ultimate_qwen_train.jsonl.gz"
    rejected_path = REPORTS / "quality_review.jsonl.gz"

    with tempfile.TemporaryDirectory(
        prefix="uq_validate_"
    ) as temp_dir:

        dedup_db = sqlite3.connect(
            str(Path(temp_dir) / "validation.sqlite")
        )

        dedup_db.execute(
            "CREATE TABLE seen (fingerprint TEXT PRIMARY KEY)"
        )

        with gzip.open(
            training_path, "wt", encoding="utf-8"
        ) as train_out, gzip.open(
            rejected_path, "wt", encoding="utf-8"
        ) as review_out:

            input_files = sorted(OUT.glob("*.jsonl.gz"))

            for path in input_files:
                logging.info("Validating: %s", path.name)

                file_source = path.name.replace(
                    ".jsonl.gz", ""
                ).replace("__", "/", 1)

                try:
                    with gzip.open(
                        path, "rt", encoding="utf-8"
                    ) as source_file:

                        for line_number, line in enumerate(
                            source_file, 1
                        ):
                            stats["read"] += 1

                            try:
                                record = json.loads(line)
                            except json.JSONDecodeError:
                                stats["invalid_json"] += 1
                                per_source[file_source]["invalid_json"] += 1
                                continue

                            if not isinstance(record, dict):
                                stats["invalid_record"] += 1
                                per_source[file_source]["invalid_record"] += 1
                                continue

                            score, reasons = inspect_sample(record)

                            metadata = record.get("metadata") or {}
                            source_name = metadata.get(
                                "source_dataset",
                                file_source
                            )

                            per_source[source_name]["read"] += 1
                            stats["scored"] += 1
                            reason_counts.update(reasons)

                            if score < 70:
                                stats["rejected_or_review"] += 1
                                per_source[source_name]["rejected_or_review"] += 1

                                review_out.write(
                                    json.dumps({
                                        "score": score,
                                        "reasons": reasons,
                                        "record": record,
                                    }, ensure_ascii=False) + "\n"
                                )
                                continue

                            messages = record.get("messages", [])

                            canonical = json.dumps(
                                messages,
                                ensure_ascii=False,
                                sort_keys=True
                            )

                            fingerprint = hashlib.sha256(
                                canonical.encode("utf-8")
                            ).hexdigest()

                            try:
                                dedup_db.execute(
                                    "INSERT INTO seen (fingerprint) VALUES (?)",
                                    (fingerprint,)
                                )
                            except sqlite3.IntegrityError:
                                stats["duplicate_in_quarantine"] += 1
                                per_source[source_name]["duplicate_in_quarantine"] += 1
                                continue

                            output_record = dict(record)
                            output_metadata = dict(metadata)

                            output_metadata["quality_screen_score"] = score
                            output_metadata["quality_screen"] = (
                                "heuristic_pass_not_fact_checked"
                            )

                            output_record["metadata"] = output_metadata

                            train_out.write(
                                json.dumps(
                                    output_record,
                                    ensure_ascii=False
                                ) + "\n"
                            )

                            stats["passed"] += 1
                            per_source[source_name]["passed"] += 1

                            if stats["read"] % 5000 == 0:
                                dedup_db.commit()
                                logging.info(
                                    "Validation progress: %d",
                                    stats["read"]
                                )

                except (OSError, EOFError, gzip.BadGzipFile) as exc:
                    stats["unreadable_gzip"] += 1
                    per_source[file_source]["unreadable_gzip"] += 1
                    logging.exception(
                        "Could not read %s: %s",
                        path.name,
                        exc
                    )

        dedup_db.commit()
        dedup_db.close()

    report = {
        "project": "ULTIMATE QWEN",
        "validation_method": "deterministic_heuristic_screening",
        "important_limitation": (
            "A pass does not guarantee factual correctness, code execution, "
            "source licensing, or absence of hidden contamination. "
            "Further review is required before training."
        ),
        "acceptance_threshold": 70,
        "statistics": dict(stats),
        "reason_counts": dict(reason_counts),
        "sources": {
            name: dict(counts)
            for name, counts in sorted(per_source.items())
        },
        "training_file": str(training_path.relative_to(ROOT)),
        "review_file": str(rejected_path.relative_to(ROOT)),
    }

    report_path = REPORTS / "validation_report.json"

    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logging.info("Quality screening complete: %s", dict(stats))

    return report_path, training_path


def create_training_zip(report_path, training_path):
    zip_path = ROOT / "ULTIMATE-QWEN-TRAINING.zip"

    with zipfile.ZipFile(
        zip_path,
        "w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=1
    ) as archive:

        archive.write(
            training_path,
            arcname="training/ultimate_qwen_train.jsonl.gz"
        )

        archive.write(
            report_path,
            arcname="reports/validation_report.json"
        )

        review_path = REPORTS / "quality_review.jsonl.gz"

        if review_path.exists():
            archive.write(
                review_path,
                arcname="reports/quality_review.jsonl.gz"
            )

        collection_report = REPORTS / "collection_report.json"

        if collection_report.exists():
            archive.write(
                collection_report,
                arcname="reports/collection_report.json"
            )

    logging.info("Created training archive: %s", zip_path)


def main():
    logging.info("ULTIMATE QWEN collection + validation started")

    dedup_db = sqlite3.connect(DB_PATH)

    dedup_db.execute(
        "CREATE TABLE IF NOT EXISTS seen "
        "(fingerprint TEXT PRIMARY KEY)"
    )

    dedup_db.commit()

    for source in SOURCES:
        collect_source(source, dedup_db)

    dedup_db.commit()
    dedup_db.close()

    collection_report = {
        "project": "ULTIMATE QWEN",
        "total_statistics": dict(collection_stats),
        "sources": source_reports,
        "errors": collection_errors,
    }

    (REPORTS / "collection_report.json").write_text(
        json.dumps(
            collection_report,
            ensure_ascii=False,
            indent=2
        ),
        encoding="utf-8"
    )

    report_path, training_path = validate_quarantine()

    create_training_zip(report_path, training_path)

    logging.info("==============================")
    logging.info("COLLECTION + VALIDATION FINISHED")
    logging.info("Read: %d", collection_stats["read"])
    logging.info("Collection accepted: %d", collection_stats["accepted"])
    logging.info("Validation report: %s", report_path)
    logging.info("Training candidate: %s", training_path)
    logging.info("==============================")

    if collection_stats["read"] == 0 and not list(OUT.glob("*.jsonl.gz")):
        raise RuntimeError(
            "No records were collected and no quarantine files exist."
        )


if __name__ == "__main__":
    main()
