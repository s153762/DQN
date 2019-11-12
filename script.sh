#!/bin/sh
#BSUB -q gpuk80
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 00:30
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err

echo "Running script..."
python3 Pong_DDQN.py
