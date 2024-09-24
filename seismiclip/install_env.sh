#!/bin/bash
# 
# Installer for package
# 
# Run: ./install_env.sh

# Create conda env
source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null || source ~/miniconda3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate seismiclip
conda env list
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
pip install --use-pep517 -e .
pip install packaging opencv-python cv2-tools torchmetrics
pip install git+https://github.com/LiyuanLucasLiu/RAdam
echo 'Created and activated environment:' $(which python)

# Check cupy works as expected
echo 'Checking torch version and GPU'
conda activate seismiclip
python -c 'import torch; print(torch.__version__);  print(torch.cuda.get_device_name(torch.cuda.current_device())); print(torch.ones(10).to("cuda:0"))'
echo 'Done!'