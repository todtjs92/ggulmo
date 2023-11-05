#!/usr/bin/bash

set -euo pipefail

# set virtual env 
source ~/venv_rec/bin/activate
#preprocess
python ./preprocess/1.raw_log.py
python ./preprocess/1.raw_meta.py
python ./preprocess/2.add_view_to_meta.py
python ./preprocess/2.user_filter.py
python ./preprocess/3.category_add_to_meta.py
python ./preprocess/3.negative_impression.py
python ./preprocess/4.make_feature_table.py

# model run and inference 
python ./models/FM/main.py





