#!/usr/bin/env bash

export CUDA_HOME=/usr/local/cuda
export DYLD_LIBRARY_PATH="$DYLD_LIBRARY_PATH:$CUDA_HOME/lib"
export PATH="$CUDA_HOME/bin:$PATH"

export DYLD_LIBRARY_PATH="$CUDA_HOME/lib"
export DYLD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib:$DYLD_LIBRARY_PATH

# echo "Running Inception Cifar10"
# python run_bottleneck.py --network inception --batch_size 32 --dataset cifar10
# echo "Running Inception Traffic"
# python run_bottleneck.py --network inception --batch_size 32 --dataset traffic
# echo "Running ResNet Cifar10"
# python run_bottleneck.py --network resnet --batch_size 32 --dataset cifar10
# echo "Running ResNet Traffic"
# python run_bottleneck.py --network resnet --batch_size 32 --dataset traffic
# echo "Running VGG Cifar10"
# python run_bottleneck.py --network vgg --batch_size 16 --dataset cifar10
# echo "Running VGG Traffic"
# python run_bottleneck.py --network vgg --batch_size 16 --dataset traffic
python3 shrink.py --network vgg --dataset traffic --size 100
python3 shrink.py --network vgg --dataset cifar10 --size 100
python3 shrink.py --network resnet --dataset traffic --size 100
python3 shrink.py --network resnet --dataset cifar10 --size 100
python3 shrink.py --network inception --dataset traffic --size 100
python3 shrink.py --network inception --dataset cifar10 --size 100