# download dataset
if test -d ../data/cifar-100-python
then
    echo "already downloaded"
else
    wget https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
    mv cifar-100-python.tar.gz ../data/
    cd ../data
    tar -xvf cifar-100-python.tar.gz
    rm cifar-100-python.tar.gz
    cd ../Q5code
fi