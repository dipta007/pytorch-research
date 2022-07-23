#!/bin/bash

echo "Run this script after activating proper conda environment"
echo "e.g. source activate pytorch"
echo "Then run this script"

read -p "Do you want to continue? (y/n) "

if [[ $REPLY =~ ^[Nn]$ ]]
then
    exit 0
fi

read -p "Do you want to install the required packages? (y/n) "

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing packages..."
    read -rp "Get the command to install PyTorch from https://pytorch.org/get-started/locally/ and paste it here: " pytorch_cmd
    echo "Running: " $pytorch_cmd
    $pytorch_cmd

    pip install wandb
    pip install hydra-core --upgrade
    pip install hydra_colorlog --upgrade
    conda install -c conda-forge tqdm
    conda install -c conda-forge prettytable
else
    echo "Skipping packages installation"
fi

read -p "Do you want pre-commit for code checking? (HIGHLY RECOMMENDED) (y/n) "

if [[ $REPLY =~ ^[Yy]$ ]]
then
    echo "Installing pre-commit..."
    pip install pre-commit
    pre-commit install
else
    echo "Skipping pre-commit installation"
fi

wandb login
