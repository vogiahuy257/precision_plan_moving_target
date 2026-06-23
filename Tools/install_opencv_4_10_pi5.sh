#!/bin/bash
set -e

OPENCV_VERSION=4.10.0
OPENCV_DIR=$HOME/opencv_src

echo "===== Install dependencies ====="
sudo apt update
sudo apt install -y \
  build-essential cmake git pkg-config \
  libgtk-3-dev \
  libavcodec-dev libavformat-dev libswscale-dev \
  libgstreamer1.0-dev libgstreamer-plugins-base1.0-dev \
  libjpeg-dev libpng-dev libtiff-dev \
  libv4l-dev v4l-utils \
  libtbb-dev \
  python3-dev python3-numpy \
  curl unzip

echo "===== Download OpenCV ${OPENCV_VERSION} ====="
mkdir -p ${OPENCV_DIR}
cd ${OPENCV_DIR}

curl -L https://github.com/opencv/opencv/archive/${OPENCV_VERSION}.zip -o opencv.zip
curl -L https://github.com/opencv/opencv_contrib/archive/${OPENCV_VERSION}.zip -o opencv_contrib.zip

unzip -q opencv.zip
unzip -q opencv_contrib.zip

cd opencv-${OPENCV_VERSION}
mkdir -p build
cd build

echo "===== Configure OpenCV ====="
cmake \
  -D CMAKE_BUILD_TYPE=Release \
  -D CMAKE_INSTALL_PREFIX=/usr/local \
  -D OPENCV_EXTRA_MODULES_PATH=${OPENCV_DIR}/opencv_contrib-${OPENCV_VERSION}/modules \
  -D WITH_GSTREAMER=ON \
  -D WITH_LIBV4L=ON \
  -D WITH_TBB=ON \
  -D BUILD_TESTS=OFF \
  -D BUILD_PERF_TESTS=OFF \
  -D BUILD_EXAMPLES=OFF \
  -D BUILD_opencv_python3=ON \
  -D OPENCV_GENERATE_PKGCONFIG=OFF \
  ..

echo "===== Build OpenCV ====="
make -j$(nproc)

echo "===== Install OpenCV ====="
sudo make install
sudo ldconfig

echo "===== Done ====="
