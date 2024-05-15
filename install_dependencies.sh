#! /bin/bash

cd ${HOME}/

if [ ! -d "Catch2" ]; then
    git clone https://github.com/catchorg/Catch2.git
    cd Catch2
    cmake -Bbuild -H. -DBUILD_TESTING=OFF
    sudo cmake --build build/ --target install
fi

if [ ! -d "eigen-3.4.0"]; then
    git clone https://gitlab.com/libeigen/eigen.git
    cd eigen-3.4.0
    mkdir build
    cd build
    cmake ..
    sudo make install
fi


