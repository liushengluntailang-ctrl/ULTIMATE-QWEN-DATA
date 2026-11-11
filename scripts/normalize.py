
import os
import json
import csv
import hashlib
import logging
from pathlib import Path
from collections import Counter

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "raw"
PROCESSED = ROOT / "processed"
REJECTED = ROOT / "rejected"
REPORTS = ROOT / "reports"

for folder in (PROCESSED, REJECTED, REPORTS):
    folder.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)

stats = Counter()
seen = set()
errors = []
categories = {
    "coding": [],
    "agent": [],
    "reasoning": [],
    "mathematics": [],
    "science": [],
    "general": []
}

def read_json(path):
    with path.open("r", encoding="utf-8-sig") as f:
        return json.load(f)

def read_jsonl(path):
    with path.open("r", encoding="utf-8-sig") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)

def read_csv(path):
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        yield from csv.DictReader(f)

def read_parquet(path):
    import pyarrow.parquet as pq
    parquet = pq.ParquetFile(path)
    for batch in parquet.iter_batches(batch_size=500):
        for row in batch.to_pylist():
            yield row

def records_from(path):
    suffix = path.suffix.lower()

    if suffix == ".jsonl":
        yield from read_jsonl(path)
    elif suffix == ".json":
        data = read_json(path)
        if isinstance(data, list):
            yield from data
        elif isinstance(data, dict):
            for key in ("data", "records", "examples", "conversations"):
                if isinstance(data.get(key), list):
                    yield from data[key]
                    return
            yield data
    elif suffix == ".csv":
        yield from read_csv(path)
    elif suffix == ".parquet":
        yield from read_parquet(path)
    else:
        stats["unsupported_files"] += 1

def get_text(value):
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(parts)
    return ""

def normalize_messages(record):
    messages = record.get("messages")

    if isinstance(messages, list):
        result = []
        for msg in messages:
            if not isinstance(msg, dict):
                continue

            role = msg.get("role") or msg.get("from")
            content = msg.get("content", msg.get("value", ""))

            if role == "human":
                role = "user"
            elif role in ("gpt", "bot"):
                role = "assistant"

            if role not in ("system", "developer", "user", "assistant", "tool"):
                continue

            normalized = {"role": role}

            if isinstance(content, (str, list)):
                normalized["content"] = content
            else:
                normalized["content"] = json.dumps(
                    content, ensure_ascii=False
                )

            for key in ("name", "tool_call_id", "tool_calls"):
                if key in msg:
                    normalized[key] = msg[key]

            result.append(normalized)

        return result

    # Common instruction/input/output format
    instruction = get_text(record.get("instruction", record.get("prompt", "")))
    input_text = get_text(record.get("input", ""))
    output = get_text(
        record.get("output", record.get("response", record.get("completion", "")))
    )

    if instruction and output:
        user_text = instruction
        if input_text:
            user_text += "\n\n" + input_text

        return [
            {"role": "user", "content": user_text},
            {"role": "assistant", "content": output}
        ]

    # Question / answer format
    question = get_text(
        record.get("question", record.get("problem", ""))
    )
    answer = get_text(
        record.get("answer", record.get("solution", ""))
    )

    if question and answer:
        return [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer}
        ]

    return None

def detect_language(messages):
    text = " ".join(
        get_text(m.get("content", ""))
        for m in messages
    )

    if not text.strip():
        return "unknown"

    import re
    japanese = len(re.findall(r"[\u3040-\u30ff\u3400-\u9fff]", text))
    latin = len(re.findall(r"[A-Za-z]", text))

    if japanese > latin:
        return "ja"
    return "en"

def detect_category(record, messages, source):
    value = " ".join([
        str(record.get("category", "")),
        str(record.get("tags", "")),
        source
    ]).lower()

    if any(x in value for x in ("agent", "tool", "trajectory")):
        return "agent"
    if any(x in value for x in ("math", "algebra", "calculus")):
        return "mathematics"
    if any(x in value for x in ("science", "physics", "biology")):
        return "science"
    if any(x in value for x in ("reason", "logic", "thinking")):
        return "reasoning"
    if any(x in value for x in ("code", "program", "debug", "software")):
        return "coding"

    content = " ".join(
        get_text(m.get("content", ""))
        for m in messages
    ).lower()

    if any(x in content for x in (
        "```", "def ", "class ", "function ", "import ", "compile"
    )):
        return "coding"

    return "general"

def process_record(record, source):
    if not isinstance(record, dict):
        stats["invalid_records"] += 1
        return

    messages = normalize_messages(record)

    if not messages or len(messages) < 2:
        stats["invalid_records"] += 1
        return

    if not any(
        m["role"] == "assistant" and get_text(m.get("content", "")).strip()
        for m in messages
    ):
        stats["invalid_records"] += 1
        return

    if detect_language(messages) != "en":
        stats["non_english"] += 1
        return

    fingerprint = hashlib.sha256(
        json.dumps(messages, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    if fingerprint in seen:
        stats["duplicates"] += 1
        return

    seen.add(fingerprint)

    category = detect_category(record, messages, source)

    output = {
        "messages": messages,
        "metadata": {
            "source_dataset": source,
            "source_id": str(
                record.get("id", record.get("source_id", ""))
            ),
            "language": "en",
            "category": category
        }
    }

    categories[category].append(output)
    stats["accepted"] += 1

def main():
    if not RAW.exists():
        logging.error("raw/ directory does not exist.")
        return

    files = [
        p for p in RAW.rglob("*")
        if p.is_file()
        and p.suffix.lower() in (".json", ".jsonl", ".csv", ".parquet")
    ]

    logging.info("Found %d source files", len(files))

    for path in files:
        source = path.relative_to(RAW).parts[0]
        logging.info("Processing: %s", path)

        try:
            for record in records_from(path):
                stats["total"] += 1
                try:
                    process_record(record, source)
                except Exception as exc:
                    stats["record_errors"] += 1
                    errors.append({
                        "file": str(path.relative_to(ROOT)),
                        "error": str(exc)
                    })
        except Exception as exc:
            stats["file_errors"] += 1
            errors.append({
                "file": str(path.relative_to(ROOT)),
                "error": str(exc)
            })
            logging.exception("Failed: %s", path)

    for category, records in categories.items():
        output_path = PROCESSED / f"{category}.jsonl"

        with output_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        logging.info("%s: %d records", category, len(records))

    report = {
        "statistics": dict(stats),
        "category_counts": {
            key: len(value) for key, value in categories.items()
        },
        "source_files": len(files)
    }

    (REPORTS / "statistics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    (REPORTS / "errors.json").write_text(
        json.dumps(errors, ensure_ascii=False, indent=2),
        encoding="utf-8"
    )

    logging.info("Normalization complete.")
    logging.info("Total: %d", stats["total"])
    logging.info("Accepted: %d", stats["accepted"])
    logging.info("Duplicates: %d", stats["duplicates"])
    logging.info("Non-English: %d", stats["non_english"])

if __name__ == "__main__":
    main()
