FROM ubuntu:22.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt update && \
    apt install -y cmake wget g++ libopencv-dev && \
    cd /root/ && \
    wget https://github.com/microsoft/onnxruntime/releases/download/v1.16.3/onnxruntime-linux-x64-1.16.3.tgz && \
    tar -zxvf onnxruntime-linux-x64-1.16.3.tgz && \
    mkdir /usr/local/include/onnxruntime && \
    mkdir /usr/local/lib/onnxruntime && \
    cp /root/onnxruntime-linux-x64-1.16.3/include/* /usr/local/include/onnxruntime/ && \
    cp /root/onnxruntime-linux-x64-1.16.3/lib/* /usr/local/lib/onnxruntime/ && \
    rm -rf /root/* 