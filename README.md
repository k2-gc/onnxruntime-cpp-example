# onnxruntime-cpp-example

## Introduction
This repository introduces how to run onnx model using onnxruntime C++ API. 
Using sample MNIST data, the code shows a simple example and inference results.

**NOTICE:** If you are a beginner in Deep Learning for image processing doamin, [my another repository](https://github.com/k2-gc/Simple-CNN-Example) may help you to understand how to train models of image classification with Pytorch.

## Prerequisites
* Docker
* Docker compose

## Usage
1. Clone this repository
2. Run commands below and get into docker container
```bash
cd onnxruntime-cpp-example
docker compose up -d
docker exec -it onnxruntime-sample /bin/bash
```
5. In docker container, build sample app and run.
```bash
mkdir build && cd build
wget https://github.com/k2-gc/Simple-CNN-Example/releases/download/v0.1/best.onnx
cmake ..
make -j$(nproc)
./sample_app 
```