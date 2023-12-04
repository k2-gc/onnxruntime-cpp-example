# onnxruntime-cpp-example

## Introduction
This repository introduces how to run onnx model using onnxruntime C++ API showing simple example.
Using sample MNIST dataset, code shows inference results.

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