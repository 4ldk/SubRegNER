#!/bin/sh

set -eux

cd subword_regularization
pip install --user -r requirements.txt
export PATH=~/.local/bin:$PATH

python3 ./bert/run.py model_name="bert-large-cased" batch_size=8 accum_iter=4 

python3 ./bert/run.py model_name="bert-large-cased" batch_size=8 accum_iter=4 p=0.1

python3 ./bert/run.py model_name="bert-large-cased" batch_size=8 accum_iter=4 P=0.3
  