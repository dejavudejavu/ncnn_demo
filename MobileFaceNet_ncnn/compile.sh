#!/bin/bash

sudo rm -rf CMakeFiles
sudo rm -f cmake_install.cmake CMakeCache.txt Makefile NCNN
cmake .
make
./NCNN