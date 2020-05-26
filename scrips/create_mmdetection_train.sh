#!/usr/bin/env bash

PYTHONPATH=/kaggle-imaterialist python /kaggle-imaterialist/src/create_mmdetection_train.py \
    --annotation=train.csv \
    --root=/train \
    --output=/data/train_mmdetection.pkl