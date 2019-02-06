#!/usr/bin/env bash
git clone https://github.com/bcamath-ds/compass
cd compass
sudo apt-get install libtool m4
sudo apt-get install libgsl0-dev libatlas-base-dev libbfd-dev libiberty-dev
sudo apt-get install libssl-dev
sudo apt-get install autoconf automake
autoheader
libtoolize
aclocal
automake --add-missing
autoconf
./configure
make
cd ..