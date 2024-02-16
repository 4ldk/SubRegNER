#!/bin/sh

set -eux

cd subword_regularization
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

python3 ./roberta/run.py model_name="roberta-large" length=510 lr=0.00001 batch_size=8 accum_iter=4

python3 ./roberta/run.py model_name="roberta-large" length=510 lr=0.00001 batch_size=8 accum_iter=4 p=0.1

python3 ./roberta/run.py model_name="roberta-large" length=510 lr=0.00001 batch_size=8 accum_iter=4 p=0.3