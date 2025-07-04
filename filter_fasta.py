#!/usr/bin/env python3

import argparse
from Bio import SeqIO
import numpy as np


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Filter sequences by length from a FASTA file."
    )
    parser.add_argument("--input", "-i", required=True, help="Input FASTA file")
    parser.add_argument(
        "--output", "-o", required=True, help="Output FASTA file for filtered sequences"
    )
    parser.add_argument(
        "--threshold",
        "-t",
        type=int,
        required=True,
        help="Length threshold (keep sequences < threshold)",
    )
    return parser.parse_args()


def compute_stats(lengths):
    print("\nSequence Length Statistics (Original Data):")
    print(f"Total Sequences: {len(lengths)}")
    print(f"Min Length     : {np.min(lengths)}")
    print(f"Max Length     : {np.max(lengths)}")
    print(f"Mean Length    : {np.mean(lengths):.2f}")
    print(f"Median Length  : {np.median(lengths)}")
    print(f"Std Dev        : {np.std(lengths):.2f}")
    print("Histogram (bins of 50):")
    hist, bins = np.histogram(lengths, bins=range(0, max(lengths) + 50, 50))
    for b, h in zip(bins, hist):
        print(f"  {b:>4}-{b + 49:<4}: {h}")


def main():
    args = parse_arguments()

    input_fasta = args.input
    output_fasta = args.output
    threshold = args.threshold

    all_records = list(SeqIO.parse(input_fasta, "fasta"))
    lengths = [len(rec.seq) for rec in all_records]
    compute_stats(lengths)

    filtered_records = [rec for rec in all_records if len(rec.seq) < threshold]
    SeqIO.write(filtered_records, output_fasta, "fasta")

    print(
        f"\nFiltered {len(filtered_records)} sequences written to '{output_fasta}' (length < {threshold})."
    )


if __name__ == "__main__":
    main()
