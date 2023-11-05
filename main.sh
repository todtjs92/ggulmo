#!/usr/bin/bash

set -euo pipefail

# set virtual env 
source ~/venv_rec/bin/activate
project_path=/home/todtjs92/ggulmo_rec/ggulmo

echo start 

#preprocess
python ${project_path}/preprocess/1.raw_log.py
python ${project_path}/preprocess/1.raw_meta.py
python ${project_path}/preprocess/2.add_view_to_meta.py
python ${project_path}/preprocess/2.user_filter.py
python ${project_path}/preprocess/3.category_add_to_meta.py
python ${project_path}/preprocess/3.negative_impression.py
python ${project_path}/preprocess/4.make_feature_table.py

# model run and inference 
python ${project_path}/models/FM/main.py





