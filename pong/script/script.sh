#!/bin/sh
#BSUB -q gpuk80
#BSUB -gpu "num=1"
#BSUB -J myJob
#BSUB -n 1
#BSUB -W 01:00
#BSUB -R "rusage[mem=32GB]"
#BSUB -o logs/%J.out
#BSUB -e logs/%J.err
#BSUB -u eida@dtu.dk
#BSUB -B
#BSUB -N

echo "Running script..."
python3 Pong_DDQN.py
echo "Script fishished."
