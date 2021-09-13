OS="ubuntu2004"
cudnn_version="8.2.4.15"
cuda_version="cuda11.4"

# cuda
# wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
# sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
# wget https://developer.download.nvidia.com/compute/cuda/11.4.2/local_installers/cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
# sudo dpkg -i cuda-repo-ubuntu2004-11-4-local_11.4.2-470.57.02-1_amd64.deb
# sudo apt-key add /var/cuda-repo-ubuntu2004-11-4-local/7fa2af80.pub
# sudo apt-get update
# sudo apt-get -y install cuda

# cuDNN
sudo apt-get install libcudnn8=${cudnn_version}-1+${cuda_version}
sudo apt-get install libcudnn8-dev=${cudnn_version}-1+${cuda_version}


## after you screw everything up and have to nuke manual installs:
#
# comm -23 \
#     <(apt-mark showmanual | sort -u) \
#     <(grep -oP '^(?:Package|Depends):\s+\K.*' /var/log/installer/status \
#         | grep -oP '[^\s,()]+?(?=(?:\s+\([^)]+\))?+(?:,|$))' \
#         | sort -u)
