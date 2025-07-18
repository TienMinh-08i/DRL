#!/bin/bash

#SBATCH --job-name=tienminh
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=gpu
#SBATCH --output=train_outs/small/out/%x.%j.out
#SBATCH --error=train_outs/small/errors/%x.%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=22010759@st.phenikaa-uni.edu.vn

conda init bash
source ~/.bashrc
conda activate env1

python main3.py