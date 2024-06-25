## Q3. LLM on Nano

### 0. Update gcc, g++ and open the fan
If you run on Ubuntu 18.04LTS or earlier, your gcc, g++ version will be too low.
Use the following command to upgrade gcc and g++ version >= 8.
<!-- ~~~bash
sudo apt-get install gcc-8 g++-8
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-8 100 --slave /usr/bin/g++ g++ /usr/bin/g++-8
~~~ -->

(Even higher version can be installed if you add another repository `sudo add-apt-repository ppa:ubuntu-toolchain-r/test`)
For example, you can install gcc-11 and g++-11 as follows:
~~~bash
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get install gcc-11 g++-11
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-11 100 --slave /usr/bin/g++ g++ /usr/bin/g++-11
# check your version.
gcc --version
~~~

~~~bash
sudo vim /sys/devices/pwm-fan/target_pwm
# insert 100 to it, and :wq close it.
~~~

### 1. Build MNN on Nano
~~~bash
git clone https://github.com/alibaba/MNN.git
~~~

If you can't git clone, download it from TA's jbox and copy to nano with a USB.

Release version, minimal verbose, used to be the underlying library.
~~~bash
mkdir release && cd release
cmake .. -DCMAKE_CXX_STANDARD=17 -DMNN_USE_SYSTEM_LIB=OFF -DMNN_BUILD_SHARED_LIBS=ON -DMNN_BUILD_TRAIN=ON -DMNN_BUILD_QUANTOOLS=ON -DMNN_EVALUATION=ON -DMNN_BUILD_CONVERTER=ON -DMNN_DEBUG_MEMORY=OFF -DMNN_DEBUG_TENSOR_SIZE=OFF -DMNN_PORTABLE_BUILD=ON -DTFMODEL_OPTIMIZE=ON -DMNN_BUILD_LLM=ON -DMNN_SUPPORT_TRANSFORMER_FUSE=ON -DMNN_LOW_MEMORY=ON
make -j4
~~~


### 2. MNN-LLM Hacking
Git clone mnn-llm.
~~~bash
git clone https://github.com/huangzhengxiang/mnn-llm.git
~~~

Soft link .so and .h from MNN.
~~~bash
ln -r -s /path/to/MNN/include/MNN include

# for lower version of ld
cp /path/to/MNN/release/libMNN.so /path/to/MNN/release/express/libMNN_Express.so libs
~~~

Then, build.
~~~bash
mkdir build && cd build
cmake .. && make -j4
cd ..
~~~

Download Qwen1.5-4B models from jbox link.
[Jbox link](https://jbox.sjtu.edu.cn/l/P1Zyow)

Put `qwen2-4b-chat` under the directory `mnn-llm/resource/models/`.

The file structure is as such:
```bash
mnn-llm
    |- ...
    |- build
    |- resource
        |- models
            |- qwen2-4b-chat
                |- block1.mnn
                |- block2.mnn
                |- ...
                |- tokenizer.txt
                |- embeddings_bf16.bin
```


Run the example.
~~~bash
cd build
./cli_demo ../resource/models/qwen2-4b-chat
~~~