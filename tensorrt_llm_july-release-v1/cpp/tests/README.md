# C++ Tests

This document explains how to build and run the C++ tests, and the included [resources](resources).

## Compile

From the top-level directory call:

```bash
CPP_BUILD_DIR=cpp/build
python3 scripts/build_wheel.py -a "80-real;86-real" --build_dir ${CPP_BUILD_DIR}
pip install -r requirements-dev.txt --extra-index-url https://pypi.ngc.nvidia.com
pip install build/tensorrt_llm*.whl
cd $CPP_BUILD_DIR && make -j$(nproc) google-tests
```

Single tests can be executed from `CPP_BUILD_DIR/tests`, e.g.

```bash
./$CPP_BUILD_DIR/tests/allocatorTest
```

## gptSessionTest

`gptSessionTest` requires to build TRT engines, which are then loaded in the test. It also requires data files which are included in [cpp/tests/resources/data](resources/data).

### Build engines

We provide a script that downloads the Huggingface GPT2 model and converts it to TRT engines.
The weights and built engines are stored under [cpp/tests/resources/models](resources/models).
To build the engines call from the top-level directory:

```bash
PYTHONPATH=examples/gpt python3 cpp/tests/resources/scripts/build_gpt_engines.py
```

### Generate expected output

The `gptSessionTest` reads inputs and expected outputs from Numpy files located under [cpp/tests/resources/data](resources/data). The expected outputs can be generated using [cpp/tests/resources/scripts/generate_expected_output.py](resources/scripts/generate_expected_output.py) which uses the built engines with the python runtime:

```bash
PYTHONPATH=examples/gpt python3 cpp/tests/resources/scripts/generate_expected_output.py
```

### Run test

After building the engines and generating the expected output execute the test

```bash
./$CPP_BUILD_DIR/tests/gptSessionTest
```

## Run all tests with ctest

To run all tests and produce an xml report, call

```bash
./$CPP_BUILD_DIR/ctest --output-on-failure --output-junit "cpp-test-report.xml"
```
