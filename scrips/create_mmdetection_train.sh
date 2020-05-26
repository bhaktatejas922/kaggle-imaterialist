#!/usr/bin/env bash

PYTHONPATH=kaggle-imaterialist python3 src/create_mmdetection_train.py \
    --annotation=train.csv \
    --output=data/train_mmdetection.pkl