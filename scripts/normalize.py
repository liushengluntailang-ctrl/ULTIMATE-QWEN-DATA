
import os
import re
import json
import gzip
import sqlite3
import hashlib
import logging
from pathlib import Path
from collections import Counter

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "quarantine"
REPORTS = ROOT / "reports"
DB_PATH = ROOT / "dedup.sqlite"

OUT.mkdir(exist_ok=True)
REPORTS.mkdir(exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

# Candidate datasets selected for ULTIMATE QWEN
SOURCES = [
    {
        "id": "AletheiaResearch/GPT-5.5-Codex",
        "category": "coding",
    },
    {
        "id": "greghavens/gpt-5.6-sol-coding-and-debugging-traces",
        "category": "agent",
    },
    {
        "id": "sollamon/ox-alpha-glm-5.3-flash-distillation-coding-17k-raw",
        "category": "coding",
    },
    {
        "id": "Crownelius/GLM-5.0-25000x",
        "category": "reasoning",
    },
    {
        "id": "saidutta69/fable-5-premium-v2",
        "category": "reasoning",
    },
    {
        "id": "DSFFGFG456/fable-5-coding-and-debugging-traces",
        "category": "agent",
    },
    {
        "id": "MoreThought/Fable-5.1-Max-Reasoning-Filtered-10000x",
        "category": "reasoning",
    },
    {
        "id": "saidutta69/claude-mythos-distilled-25k-clean",
        "category": "reasoning",
    },
    {
        "id": "faunix/Qwen3.8-27B-Distillation-40K",
        "category": "general",
    },
    {
        "id": "Jackrong/DeepSeek-V4-Distill-8000x",
        "category": "reasoning",
    },
]

stats = Counter()
errors = []
source_reports = []

db = sqlite3.connect(DB_PATH)
db.execute("""
    CREATE TABLE IF NOT EXISTS seen (
        fingerprint TEXT PRIMARY KEY
    )
""")
db.commit()


def text_value(value):
    if isinstance(value, str):
        return value

    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                if isinstance(item.get("text"), str):
                    parts.append(item["text"])
        return "\n".join(parts)

    if value is None:
        return ""

    return json.dumps(value, ensure_ascii=False)


def normalize_messages(record):
    messages = record.get("messages")

    if not isinstance(messages, list):
        messages = record.get("conversations")

    if isinstance(messages, list):
        result = []

        for item in messages:
            if not isinstance(item, dict):
                continue

            role = (
                item.get("role")
                or item.get("from")
                or item.get("speaker")
            )

            role = str(role).lower() if role else ""

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

            role = role_map.get(role)

            if not role:
                continue

            content = item.get(
                "content",
                item.get("value", item.get("text", ""))
            )

            message = {
                "role": role,
                "content": content if isinstance(
                    content, (str, list, dict)
                ) else text_value(content)
            }

            for key in (
                "name",
                "tool_call_id",
                "tool_calls",
                "function_call"
            ):
                if key in item:
                    message[key] = item[key]

            result.append(message)

        if result:
            return result

    # Instruction / input / output datasets
    instruction = text_value(
        record.get("instruction", record.get("prompt", ""))
    )

    extra_input = text_value(record.get("input", ""))

    answer = text_value(
        record.get(
            "output",
            record.get(
                "response",
                record.get(
                    "completion",
                    record.get("chosen", "")
                )
            )
        )
    )

    if instruction and answer:
        user_text = instruction

        if extra_input:
            user_text += "\n\n" + extra_input

        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": answer}
        ]

    # Question / answer datasets
    question = text_value(
        record.get("question", record.get("problem", ""))
    )

    answer = text_value(
        record.get("answer", record.get("solution", ""))
    )

    if question and answer:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

    return None


def english_check(messages, record):
    declared = str(
        record.get("language", record.get("lang", ""))
    ).lower()

    if declared:
        if declared in ("en", "eng", "english"):
            pass
        elif declared in ("ja", "jp", "japanese", "zh", "chinese"):
            return False

    content = " ".join(
        text_value(m.get("content", ""))
        for m in messages
    )

    # Exclude Japanese-script examples.
    if re.search(r"[\u3040-\u30ff\u31f0-\u31ff]", content):
        return False

    letters = re.findall(r"[A-Za-z]", content)
    non_latin = re.findall(
        r"[\u4e00-\u9fff\u0400-\u04ff\u0600-\u06ff]",
        content
    )

    if letters and len(non_latin) / len(letters) > 0.20:
        return False

    return bool(content.strip())


def make_record(row, source, index):
    messages = normalize_messages(row)

    if not messages:
        return None

    roles = {m["role"] for m in messages}

    if "user" not in roles or "assistant" not in roles:
        return None

    if not english_check(messages, row):
        stats["non_english"] += 1
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
        db.execute(
            "INSERT INTO seen (fingerprint) VALUES (?)",
            (fingerprint,)
        )
    except sqlite3.IntegrityError:
        stats["duplicates"] += 1
        return None

    metadata = {
        "source_dataset": source["id"],
        "source_id": str(row.get("id", index)),
        "category": source["category"],
        "language": "en",
        "rights_status": "unverified_quarantine"
    }

    for key in ("teacher_model", "model", "difficulty"):
        if row.get(key) is not None:
            metadata[key] = row[key]

    return {
        "messages": messages,
        "metadata": metadata
    }


def get_splits(dataset):
    if hasattr(dataset, "keys"):
        return [
            (str(name), split)
            for name, split in dataset.items()
        ]

    return [("train", dataset)]


def process_source(source):
    repo = source["id"]
    logging.info("Starting: %s", repo)

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
                logging.info(
                    "%s: processing split %s",
                    repo,
                    split_name
                )

                for index, row in enumerate(split):
                    source_stats["read"] += 1
                    stats["read"] += 1

                    try:
                        item = make_record(row, source, index)

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
                        stats["accepted"] += 1

                        if stats["read"] % 5000 == 0:
                            db.commit()
                            logging.info(
                                "Progress: %s rows",
                                stats["read"]
                            )

                    except Exception as exc:
                        source_stats["errors"] += 1
                        errors.append({
                            "dataset": repo,
                            "row": index,
                            "error": str(exc)
                        })

        db.commit()

        result = {
            "dataset": repo,
            "file": str(output_path.relative_to(ROOT)),
            "status": "success",
            "statistics": dict(source_stats)
        }

        logging.info(
            "Finished %s: %s",
            repo,
            dict(source_stats)
        )

    except Exception as exc:
        db.commit()

        result = {
            "dataset": repo,
            "status": "failed",
            "error": str(exc)
        }

        errors.append(result)
        logging.exception("Dataset failed: %s", repo)

    source_reports.append(result)


def main():
    logging.info(
        "ULTIMATE QWEN collection started. Sources: %d",
        len(SOURCES)
    )

    for source in SOURCES:
        process_source(source)

    db.commit()
    db.close()

    report = {
        "project": "ULTIMATE QWEN",
        "total_statistics": dict(stats),
        "sources": source_reports,
        "errors": errors
    }

    (REPORTS / "collection_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logging.info("================================")
    logging.info("COLLECTION FINISHED")
    logging.info("Read: %d", stats["read"])
    logging.info("Accepted: %d", stats["accepted"])
    logging.info("Duplicates: %d", stats["duplicates"])
    logging.info("Non-English: %d", stats["non_english"])
    logging.info("================================")

    if stats["read"] == 0:
        raise RuntimeError(
            "No records were read from any dataset."
        )


if __name__ == "__main__":
    main()
