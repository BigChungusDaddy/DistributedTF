apt update
apt-get update
apt -y install build-essential zlib1g-dev \
libncurses5-dev libgdbm-dev libnss3-dev \
libssl-dev libreadline-dev libffi-dev curl software-properties-common
wget https://www.python.org/ftp/python/3.9.15/Python-3.9.15.tar.xz
tar -xf Python-3.9.15.tar.xz
cd Python-3.9.15
./configure
make altinstall
cd ..