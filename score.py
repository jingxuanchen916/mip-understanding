"""Accuracy tables from run.py output.

    python score.py results/*.jsonl
    python score.py results/*.jsonl --by task n_instances

accuracy counts an invalid response (no parseable answer after 3 attempts) as wrong;
valid_accuracy is over valid responses only.
"""

import argparse
import json

import pandas as pd

GROUP_COLUMNS = ("task", "n_instances", "selection_seed")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("files", nargs="+")
    ap.add_argument("--by", nargs="*", default=["task"], choices=GROUP_COLUMNS,
                    help="columns to break results down by, after model (default: task)")
    args = ap.parse_args()

    records = []
    for path in args.files:
        with open(path) as f:
            for line in f:
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass  # a line cut off when a run was killed
    df = pd.DataFrame(records)
    df = df.drop_duplicates(["model", "id"], keep="last")  # a row run twice counts once

    keys = ["model", *args.by]
    table = df.groupby(keys).agg(
        n=("correct", "size"),
        accuracy=("correct", "mean"),
        valid=("valid", "mean"),
    )
    table["valid_accuracy"] = df[df["valid"]].groupby(keys)["correct"].mean()
    overall = df.groupby(keys[:1]).agg(n=("correct", "size"), accuracy=("correct", "mean"),
                                       valid=("valid", "mean"))
    overall["valid_accuracy"] = df[df["valid"]].groupby(keys[:1])["correct"].mean()

    pct = {c: "{:.1%}".format for c in ("accuracy", "valid", "valid_accuracy")}
    if args.by:
        print(table.to_string(formatters=pct))
        print()
    print(overall.to_string(formatters=pct))


if __name__ == "__main__":
    main()
