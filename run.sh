#!/bin/bash

# Activate the mamba environment named 'wj'

echo "Activating venv..."

# source .venv/bin/activate
source $(conda info --base)/etc/profile.d/conda.sh

conda activate mdlm

echo "Running training script in the background..."

# nohup python3 /home/dhruva/discrete-diffusion-guidance/main.py > ./nohup/run3.out 2>&1 &
nohup python3 /home/dhruva/discrete-diffusion-guidance/sample_sequences_v2.py > ./nohup/new_model_100k.out 2>&1 &

