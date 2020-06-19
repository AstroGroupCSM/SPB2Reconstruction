#!/bin/bash

MODEL_TIME_ID=$1

export PRED_FILE="$PWD/saved/$MODEL_TIME_ID"

cd code

python3 evaluate_predictions.py --preds "$PRED_FILE"

echo "** Compressing dir **"
cd ../saved

tar -czf "$MODEL_TIME_ID.tar.gz" --exclude models "$MODEL_TIME_ID"