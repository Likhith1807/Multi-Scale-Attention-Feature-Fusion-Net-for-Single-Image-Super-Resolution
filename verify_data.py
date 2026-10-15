"""Verify all 900 official DIV2K x4 pairs without downloading or modifying images."""
import argparse
from pathlib import Path

from evaluate import save_json
from pairs import find_pairs, check_disjoint, read_pair, fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--output", default="runs/dataset_verification.json")
    args = parser.parse_args()
    root = Path(args.data)
    groups = {}
    for split, expected, first in (("train", 800, 1), ("valid", 100, 801)):
        pairs = find_pairs(root / f"DIV2K_{split}_LR_bicubic/X4",
                           root / f"DIV2K_{split}_HR", scale=4)
        expected_ids = {f"{i:04d}" for i in range(first, first + expected)}
        if {pair.name for pair in pairs} != expected_ids:
            raise ValueError(f"{split}: expected the official {expected} source IDs starting at {first:04d}.")
        for index, pair in enumerate(pairs, 1):
            low, high = read_pair(pair, 4)
            if min(low.shape[:2]) < 48:
                raise ValueError(f"{pair.name}: too small for the default training crop")
            if index % 100 == 0:
                print(f"Decoded and checked {split}: {index}/{expected}", flush=True)
        groups[split] = pairs
    check_disjoint(groups["train"], groups["valid"])
    print("Hashing image files for the dataset record...", flush=True)
    manifests = {name: fingerprint(pairs) for name, pairs in groups.items()}
    hashes = {row["hr_sha256"] for row in manifests["train"]["images"]}
    if hashes & {row["hr_sha256"] for row in manifests["valid"]["images"]}:
        raise ValueError("Byte-identical HR files exist across training and validation")
    save_json(args.output, {"status": "passed", "scale": 4, "manifests": manifests,
                           "checks": "Official IDs, decode, dimensions, disjoint IDs/paths and HR file hashes",
                           "limitation": "Not an authenticity check against publisher checksums; dimensions do not prove pixel registration."})
    print("Dataset verification passed: 800 training pairs and 100 validation pairs.")
    print(f"Report saved to {args.output}")


if __name__ == "__main__":
    main()
