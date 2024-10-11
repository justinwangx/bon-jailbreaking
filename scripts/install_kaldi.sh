sudo apt-get update
sudo apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    make \
    automake \
    bzip2 \
    unzip \
    wget \
    sox \
    libtool \
    git \
    subversion \
    python2.7 \
    python3 \
    zlib1g-dev \
    ca-certificates \
    gfortran \
    patch \
    ffmpeg \
    vim
sudo apt-get update
sudo apt-get install -y --no-install-recommends software-properties-common
apt-add-repository multiverse
sudo apt-get update
sudo apt-get install -yqq --no-install-recommends intel-mkl

sudo ln -s /usr/bin/python2.7 /usr/bin/python

sudo git clone --depth 1 https://github.com/kaldi-asr/kaldi.git /opt/kaldi
sudo chmod -R 777 /opt/kaldi
cd /opt/kaldi/tools
make -j $(nproc)
cd /opt/kaldi/src
./configure --shared
make depend -j $(nproc)
make -j $(nproc)
find /opt/kaldi  -type f \( -name "*.o" -o -name "*.la" -o -name "*.a" \) -exec rm {} \;

# To install full RIRs and MUSAN
cd ./data
wget https://us.openslr.org/resources/17/musan.tar.gz
wget --no-check-certificate http://www.openslr.org/resources/28/rirs_noises.zip
tar -xzf musan.tar.gz
unzip rirs_noises.zip
cd ..