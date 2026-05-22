#!/bin/bash

# 清华镜像站 PyPI 下载批处理脚本
# 用法: bash tsinghua_wget_script.sh

# 定义要下载的文件列表
files=(
    "https://pypi.tuna.tsinghua.edu.cn/packages/source/t/tensorflow/tensorflow-2.14.0.tar.gz"
    "https://pypi.tuna.tsinghua.edu.cn/packages/source/t/torch/torch-2.1.0.tar.gz"
    "https://pypi.tuna.tsinghua.edu.cn/packages/source/t/torchvision/torchvision-0.15.2.tar.gz"
    "https://pypi.tuna.tsinghua.edu.cn/packages/source/t/torchaudio/torchaudio-2.1.0.tar.gz"
    "https://pypi.tuna.tsinghua.edu.cn/packages/source/t/transformers/transformers-4.41.0.tar.gz"
)

# 下载目录
download_dir="tsinghua_packages"
mkdir -p "$download_dir"

# 循环下载
for url in "${files[@]}"; do
    wget -c "$url" -P "$download_dir"
    echo "Downloaded $url"
    sleep 1
 done

echo "All downloads completed."