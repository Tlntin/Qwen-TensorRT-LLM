FROM nvcr.io/nvidia/pytorch:23.04-py3

# update timezone
ENV TZ=Asia/Shanghai
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone && \
    apt update && \
    apt upgrade -y && \
    apt install git tzdata libgl1-mesa-glx -y
RUN dpkg-reconfigure --frontend noninteractive tzdata

# Python packages
COPY requirements-dev.txt /tmp/
RUN pip install -r /tmp/requirements-dev.txt

# Remove prevous TRT installation
RUN apt-get remove --purge -y libnvinfer* tensorrt*
RUN pip uninstall -y tensorrt

# Download or copy from local file
# Reference: https://github.com/NVIDIA/TensorRT/tree/release/9.0
# RUN wget https://developer.nvidia.com/downloads/compute/machine-learning/tensorrt/secure/9.0.1/tars/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -P /workspace
COPY TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz /workspace/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz

# install TensorRT
RUN tar -xvf /workspace/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz -C /usr/local/ && mv /usr/local/TensorRT-9.0.1.4 /usr/local/TensorRT
RUN pip install /usr/local/TensorRT/python/tensorrt-9.0.1*cp38-none-linux_x86_64.whl && rm /workspace/TensorRT-9.0.1.4.Linux.x86_64-gnu.cuda-12.2.tar.gz
ENV LD_LIBRARY_PATH=/usr/local/TensorRT/lib/:/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu/:/usr/local/cuda/lib64/:$LD_LIBRARY_PATH

# update ldconfig
RUN echo "/usr/local/tensorrt/lib" >> /etc/ld.so.conf.d/tensorrt.conf
RUN ldconfig -v && ldconfig -p

# Install Polygraphy v0.48.1.
RUN python -m pip install colored polygraphy==0.48.1 --extra-index-url https://pypi.ngc.nvidia.com

# set workdir
USER root
WORKDIR /root/workspace

# copy trt-llm to workspace
COPY tensorrt_llm_july-release-v1 /root/workspace/tensorrt_llm

# link libcuda (for build trt-llm)
RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/local/cuda/lib64/stubs/libcuda.so.1
 
# build trt-llm cpp
RUN mkdir /root/workspace/tensorrt_llm/cpp/build && \
    cd /root/workspace/tensorrt_llm/cpp/build && \
    cmake .. \
    -D TRT_INCLUDE_DIR=/usr/local/TensorRT/include \
    -D nvinfer_LIB_PATH=/usr/local/TensorRT/targets/x86_64-linux-gnu/lib/libnvinfer.so \
    -D nvonnxparser_LIB_PATH=/usr/local/TensorRT/targets/x86_64-linux-gnu/lib/libnvonnxparser.so \
    -D CMAKE_BUILD_TYPE=Release && \
    make -j$(nproc)
 
# build trt-llm python
RUN touch /root/workspace/tensorrt_llm/3rdparty/cutlass/.git && \
    cd /root/workspace/tensorrt_llm/scripts && \
    python build_wheel.py
 
# pip install trt-llm python
RUN cd /root/workspace/tensorrt_llm/build && \
    pip install tensorrt_llm-*.whl

# rm libcuda
RUN rm /usr/local/cuda/lib64/stubs/libcuda.so.1