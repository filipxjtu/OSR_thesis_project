from __future__ import annotations

import sys

from python.src.validation.gate import run_validation_gate
from python.src.validation.exceptions import ValidationError


def main():

    CLEAN_FILE = "clean_dataset_v1_seed8.mat"
    TRAIN_FILE = "impaired_dataset_v1_seed8_train.mat"
    EVAL_FILE  = "impaired_dataset_v1_seed8_eval.mat"

    try:
        run_validation_gate(
            clean_file=CLEAN_FILE,
            train_file=TRAIN_FILE,
            eval_file=EVAL_FILE,
            spec_version="v1",
            n_classes=7,
            report_name="validation_seed49.json",
        )
    except ValidationError as e:
        print("\n❌ DATASET VALIDATION FAILED")
        print(e)
        sys.exit(1)

    print("\n✅ DATASET VALIDATION PASSED")


if __name__ == "__main__":
    main()