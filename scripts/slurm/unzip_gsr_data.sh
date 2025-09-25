#!/bin/bash
#SBATCH --job-name=unzip_gsr_data
#SBATCH --partition=dgpu
#SBATCH --account=researchers
#SBATCH --output=logs/unzip_gsr_data_%j.out
#SBATCH --error=logs/unzip_gsr_data_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=2

cd /home/mseo/CornerTactics/sn-gamestate/data/SoccerNetGS

unzip gamestate-2024/train.zip -d train
unzip gamestate-2024/valid.zip -d valid
unzip gamestate-2024/test.zip -d test
unzip gamestate-2024/challenge.zip -d challenge