"""Convert the v6 list CSV (Origin, Original_Sentence, Generated_Sentence, ...)
into the (sentence1, sentence2, label) format consumed by
BERT/run_glue_no_trainer.py, with an 80/20 train/validation split.

Mirrors legacy/amr_lda/train_validation_split.py but skips the deprecated
DataFrame.append usage and is parametrised on input/output.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--input-csv", type=Path,
        default=Path("legacy/data/Synthetic_xfm_t5wtense_logical_equivalence_list_v6.csv"),
    )
    ap.add_argument(
        "--train-csv", type=Path,
        default=Path("output_result/Synthetic_xfm_t5wtense_logical_equivalence_train_v6.csv"),
    )
    ap.add_argument(
        "--val-csv", type=Path,
        default=Path("output_result/Synthetic_xfm_t5wtense_logical_equivalence_validation_v6.csv"),
    )
    ap.add_argument("--train-frac", type=float, default=0.8)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    df = df.dropna(subset=["Original_Sentence", "Generated_Sentence", "Label"])
    df = df[df["Generated_Sentence"].astype(str).str.strip() != ""]
    df = df.rename(
        columns={"Original_Sentence": "sentence1",
                 "Generated_Sentence": "sentence2",
                 "Label": "label"}
    )[["sentence1", "sentence2", "label"]]
    df = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

    n_train = int(len(df) * args.train_frac)
    train = df.iloc[:n_train]
    val = df.iloc[n_train:]

    args.train_csv.parent.mkdir(parents=True, exist_ok=True)
    args.val_csv.parent.mkdir(parents=True, exist_ok=True)
    train.to_csv(args.train_csv, index=False, encoding="utf8")
    val.to_csv(args.val_csv, index=False, encoding="utf8")
    print(f"train {len(train)} -> {args.train_csv}")
    print(f"val   {len(val)} -> {args.val_csv}")
    print(f"label dist (train): {dict(train['label'].value_counts())}")
    print(f"label dist (val):   {dict(val['label'].value_counts())}")


if __name__ == "__main__":
    main()
