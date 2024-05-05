#!/bin/bash

###OCELOTE
 
#SBATCH --job-name=vitsesperanto
#SBATCH --output=%x-%j.out
#SBATCH --account=ua-hlt
#SBATCH --partition=standard
#SBATCH --ntasks=28
#SBATCH --nodes=1
#SBATCH --mem-per-cpu=8gb
#SBATCH --gres=gpu:1
#SBATCH --time=80:00:00
 
IMAGE=/home/u6/pbarrett520/coqui-tts2.sif

echo "starting"
SINGULARITYENV_CUDA_VISIBLE_DEVICES=0 singularity exec \
	--nv $IMAGE python3 /home/u6/pbarrett520/vits/no_punct_vitsesperanto.py
echo "finished"

