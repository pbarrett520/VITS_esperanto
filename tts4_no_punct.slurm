#!/bin/bash

#SBATCH --job-name=tts4_no_punct
#SBATCH --output=%x-%j.out
#SBATCH --account=ua-hlt
#SBATCH --partition=standard
#SBATCH --ntasks=14
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=6gb
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00

IMAGE=/home/u6/pbarrett520/coqui-tts2.sif

echo "starting"

singularity exec --nv $IMAGE tts --text "En la komenco Dio kreis la cxielon kaj la teron. Kaj Dio diris: Estu lumo; kaj farigxis lumo." \
  --model_path /home/u6/pbarrett520/vits/vits_esperanto-February-21-2024_05+59PM-0000000/best_model.pth \
  --config_path /home/u6/pbarrett520/vits/vits_esperanto-February-21-2024_05+59PM-0000000/config.json \
  --out_path esperanto_no_punct.wav

echo "finished"

