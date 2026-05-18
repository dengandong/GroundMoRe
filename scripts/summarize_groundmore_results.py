#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from pathlib import Path


SPLIT_ORDER = ["Overall", "Causal", "Sequential", "Counterfactual", "Descriptive"]


def load_qtype_map(meta_file):
    with open(meta_file, "r") as f:
        videos = json.load(f)["videos"]

    qtypes = {}
    for video_id, video_meta in videos.items():
        for exp_id, exp_meta in video_meta["questions"].items():
            qtypes[(video_id, str(exp_id))] = exp_meta["q_type"]
    return qtypes


def summarize(result_file, meta_file):
    qtypes = load_qtype_map(meta_file)
    buckets = defaultdict(list)

    with open(result_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = [part.strip() for part in line.split(",")]
            if len(parts) != 5:
                continue
            video_id, exp_id, jf, j, f_score = parts
            row = (float(jf), float(j), float(f_score))
            buckets["Overall"].append(row)
            qtype = qtypes.get((video_id, exp_id))
            if qtype:
                buckets[qtype].append(row)

    rows = []
    for split in SPLIT_ORDER:
        values = buckets.get(split, [])
        if not values:
            rows.append((split, 0, float("nan"), float("nan"), float("nan")))
            continue
        n = len(values)
        jf = sum(v[0] for v in values) / n
        j = sum(v[1] for v in values) / n
        f_score = sum(v[2] for v in values) / n
        rows.append((split, n, jf, j, f_score))
    return rows


def format_table(rows):
    lines = [
        "| Split          | N    | JF     | J      | F      |",
        "|----------------|------|--------|--------|--------|",
    ]
    for split, n, jf, j, f_score in rows:
        lines.append(f"| {split:<14} | {n:<4} | {jf:.4f} | {j:.4f} | {f_score:.4f} |")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--meta-file", default="groundmore_test.json")
    parser.add_argument("--result-file", action="append", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--checkpoint", action="append", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="GroundMoRe Results")
    parser.add_argument("--training-setting", default="")
    parser.add_argument("--evaluation-setting", default="")
    args = parser.parse_args()

    if not (
        len(args.result_file) == len(args.label) == len(args.checkpoint)
    ):
        raise ValueError("--result-file, --label, and --checkpoint counts must match")

    out = [args.title, ""]
    if args.training_setting:
        out.extend(["Training setting:", args.training_setting.strip(), ""])
    if args.evaluation_setting:
        out.extend(["Evaluation setting:", args.evaluation_setting.strip(), ""])

    for label, checkpoint, result_file in zip(
        args.label, args.checkpoint, args.result_file
    ):
        result_path = Path(result_file)
        if not result_path.exists():
            raise FileNotFoundError(result_file)
        rows = summarize(result_path, args.meta_file)
        out.extend(
            [
                f"{label}:",
                f"- Checkpoint: {checkpoint}",
                f"- Result file: {result_file}",
                "",
                format_table(rows),
                "",
            ]
        )

    Path(args.output).write_text("\n".join(out).rstrip() + "\n")


if __name__ == "__main__":
    main()
