#!/usr/bin/bash

set -euo pipefail

# set virtual env 
venv_python=/home/todtjs92/venv_rec/bin/python
project_path=/home/todtjs92/ggulmo_rec/ggulmo

echo start 

#preprocess
${venv_python} ${project_path}/preprocess/1.raw_log.py
${venv_python} ${project_path}/preprocess/1.raw_meta.py
${venv_python} ${project_path}/preprocess/2.add_view_to_meta.py
${venv_python} ${project_path}/preprocess/2.user_filter.py
${venv_python} ${project_path}/preprocess/3.category_add_to_meta.py
${venv_python} ${project_path}/preprocess/3.negative_impression.py
${venv_python} ${project_path}/preprocess/4.make_feature_table.py

# model run and inference 
${venv_python} ${project_path}/models/FM/main.py





