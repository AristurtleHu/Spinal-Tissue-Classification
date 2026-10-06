#!/usr/bin/env bash
# Run the complete Project 1 pipeline; results land in code/results/.
set -e
cd "$(dirname "$0")"
python common.py
python features.py
python task1_rule.py
python task2_ml.py
python aggregation_study.py
python task3_vit.py
