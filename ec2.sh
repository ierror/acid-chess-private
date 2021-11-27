#!/bin/bash

SERVER="18.134.240.200"
SSH_CMD="ssh -i ~/.ssh/aws-ec2-bernhard-london.pem ubuntu@$SERVER"

run_command () {
  $SSH_CMD "$1"
}

#run_command "wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin"
#run_command "sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600"
#run_command "sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/7fa2af80.pub"
#run_command "sudo add-apt-repository \"deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /\""
#run_command "sudo apt-get update"
#run_command "sudo apt -y install cuda"
#
#rsync -va  --exclude '*boards.model*' --exclude 'tmp/*' --exclude '*.mov' --exclude '*.model' --exclude '*-checkpoint.ipynb' -e "ssh -i ~/.ssh/bernhard-aws.pem" . ubuntu@$SERVER:/home/ubuntu/
#
#run_command "sudo apt -y install python3-pip"
#run_command "pip3 install pipenv importlib-resources"
#run_command "pip3 install --upgrade torch==1.10.0+cu113 torchvision==0.11.1+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html"
#run_command "/home/ubuntu/.local/bin/pipenv install --dev --system"

$SSH_CMD -L 127.0.0.1:8001:127.0.0.1:8888 /home/ubuntu/.local/bin/jupyter-lab
