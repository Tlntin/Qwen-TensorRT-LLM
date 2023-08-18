#!/usr/bin/bash

set -e

BUILD_DIR="cpp/build"
LIB_DIR="tensorrt_llm/libs"

# Loop through arguments and process them
for arg in "$@"
do
    case $arg in
        --clean)
        echo "[Build Cache] Wiping build directory: \`${BUILD_DIR}\`..."
        rm -rf ${BUILD_DIR}
        shift # Remove --clean from processing
        ;;
        *)
        echo "Unknown argument received: \`${arg}\`"
        exit 1
        ;;
    esac
done

mkdir -p ${BUILD_DIR}
pushd ${BUILD_DIR}

export LD_LIBRARY_PATH="/usr/local/cuda/lib64/stubs:/usr/lib/x86_64-linux-gnu/:${LD_LIBRARY_PATH}"

# Make sure to rebuild from scratch
rm -rf "${BUILD_DIR}/tensorrt_llm/plugins"

cmake \
    -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_PYT=OFF \
    -DTRT_LIB_DIR=/usr/lib/x86_64-linux-gnu/ \
    -DNCCL_LIB_DIR=/usr/lib/x86_64-linux-gnu/ \
    -DCUDNN_ROOT_DIR=/usr/lib/x86_64-linux-gnu/ \
    ..

make -j"$(grep -c ^processor /proc/cpuinfo)" tensorrt_llm tensorrt_llm_static nvinfer_plugin

popd

# copy built lib
rm -rf ${LIB_DIR}
mkdir -p ${LIB_DIR}

cp \
    "${BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin.so" \
    "${LIB_DIR}/libnvinfer_plugin.so"

cp \
    "${BUILD_DIR}/tensorrt_llm/plugins/libnvinfer_plugin.so" \
    "examples/cpp_library/libtensorrt_llm_plugin.so"
