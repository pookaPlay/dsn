# Installing instructions

## OS details
1. OS: Netrunner 20.01 - Twenty ( based on Debian-10 (or) Buster)
2. Arch: amd64
3. Date: June 2nd 2020

---
## Required libraries
### Boost library
```bash
sudo apt-get install libboost-all-dev 
```

The following code requires boost library 1.65.1. This is not availabe from
standard repository at this time. To install this library the instructions
are as follows [Link](https://stackoverflow.com/questions/8430332/uninstall-boost-and-install-another-version),
### 1. Remove all libraries installed by system
```bash
sudo apt-get update
# to uninstall deb version
sudo apt-get -y --purge remove libboost-all-dev libboost-doc libboost-dev
# to uninstall the version which we installed from source
sudo rm -f /usr/lib/libboost_*
```
#### 2. install other dependencies
```bash
sudo apt-get -y install build-essential g++ python-dev autotools-dev libicu-dev libbz2-dev
```
#### 3. Download, build and install
```bash
wget https://sourceforge.net/projects/boost/files/boost/1.65.1/boost_1_65_1.tar.gz
tar -xvf tar -xvf boost_1_65_1.tar.gz
cd boost_1_65_1
echo "Available CPU cores: "$cpuCores
./bootstrap.sh  # this will generate ./b2
sudo ./b2 --with=all -j $cpuCores install
```
#### 4. Check
```bash
cat /usr/local/include/boost/version.hpp | grep "BOOST_LIB_VERSION"
```
### OpenCV
```bash
sudo apt-get install libopencv-dev
```

### gdal

```
sudo apt-get install gdal-bin libgdal-dev
```
### log4cplus
```bash
sudo apt install liblog4cplus-dev
```
### cfitsio
```bash
sudo apt-get install libcfitsio-dev
```

## Installation
In debian,
```bash
# from dsn/src/dsnCpp/
mkdir build
cd build
cmake ../
make
```
This will compile and store `lib` and `bin` files at `dsn/src/dsnCpp/build/install`.
