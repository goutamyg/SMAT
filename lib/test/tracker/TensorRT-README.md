# Tracker Model Conversion & Inference - ONNX to TensorRT (FP32 & INT8)

## NVIDIA TensorRT

NVIDIA® TensorRT™, an SDK for high-performance deep learning inference, includes a deep learning inference optimizer and runtime that delivers low latency and high throughput for inference applications.

A few key features and components:

- Speed Up Inference: NVIDIA TensorRT based applications perform upto 36X faster than CPU-only platforms during inference. It supports a variety of major frameworks such as TEnsorFlow, PyTorch, etc.

- Optimize Inference Performance: TensorRT, built on the NVIDIA CUDA® parallel programming model, enables you to optimize inference using techniques such as quantization, layer and tensor fusion, kernel tuning, and others on NVIDIA GPUs.

- Support for Different Workloads: TensorRT provides INT8 using quantization-aware training and post-training quantization and floating point 16 (FP16) optimizations for deployment of deep learning inference applications. Reduced-precision inference significantly minimizes latency, which is required for many real-time services, as well as autonomous and embedded applications.

## Installation

To install NVIDIA TensorRT, please refer to their [official website](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) for a step-by-step process based on your hardware and operating system. You can choose between the following installation options when installing TensorRT; Debian or RPM packages, a Python wheel file, a tar file, or a zip file.

Ensure you have the following installation requirements:
- CUDA-Python
- NVIDIA CUDA™ 11.0 - 12.1
- cuDNN
- TensorFlow (optional)
- PyTorch 1.13.1 and above
- ONNX 1.12.0 and above (tested with opset 16)

The Debian and RPM installations automatically install any dependencies, however, it:
- requires sudo or root privileges to install
- provides no flexibility as to which location TensorRT is installed into
- requires that the CUDA Toolkit and cuDNN have also been installed using Debian or RPM packages.
- does not allow more than one minor version of TensorRT to be installed at the same time
- additionally, you must manage the ```LD_LIBRARY_PATH```

I have employed the Python Package Index Installation route. All required libraries are included in the Python packgae. The ```TensorRT``` Python wheel only supports Python versions 3.6 to 3.11 at this time and will not work with other Python versions. Only the Linux operating system and x86_64 CPU architecture is currently supported.

It has two Runtime APIs i.e., Python and C++ API. For the scope of this project, I have worked with the Python API found [here](https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/index.html). I have outlined the steps needed to install TensorRT Python API.

- Step#1: Create a conda virtual environment. 
```
conda env create --prefix /speed-scratch/$USER/mobilevit-track -f smat_pyenv.yml
```
- Step#2: Activate virtual environment
```
conda activate /speed-scratch/$USER/mobilevit-track
```
Note: Steps 1-2 can be skipped if you already have a virtual environment.

- Step#3: Upgrade pip to latest version
```
python -m pip install --upgrade pip
``` 
- Step#4: Download and install the package
```
python3 -m pip install --upgrade tensorrt
```
The above pip command will pull in all the required CUDA libraries and cuDNN in Python wheel format from PyPI because they are dependencies of the TensorRT Python wheel. Also, it will upgrade tensorrt to the latest version if you had a previous version installed.

- Step#5: To verify the installation is working, use the following Python commands:
```
python3
>>> import tensorrt
>>> print(tensorrt.__version__)
>>> assert tensorrt.Builder(tensorrt.Logger())
```
- Step#6: Set the ```LD_LIBRARY_PATH``` so that TensorRT functions properly:
```
setenv LD_LIBRARY_PATH /encs/pkg/anaconda3-version/root/lib:/speed-scratch/$USER/mobilevit-track/lib/python-version/site-packages/torch_tensorrt
```

I have worked with SPEED Cluster nodes for this particular task. SPEED High Performance Clusters are available for use and support GPU devices, which is a requirement to leverage TensorRT and CUDA programming. Please refer to thier [official github reporsitory](https://github.com/NAG-DevOps/speed-hpc/tree/slurm/src) for detailed overview of the facility and its usage.  


## Hardware Specifications

### CPU Specifications
- Processor:  Intel(R) Xeon(R) Gold 6130 CPU @ 2.10GHz
- RAM: 12.0 GB 
- System Type: x86_64 GNU Linux, 32-bit, 64-bit

### Speed Node Specifications
- Linux speed-01.encs.concordia.ca
- GPU: Tesla P6
- VRAM: 16.0 GB

## TensorRT Workflow

We assume that one has a trained model to begin with. You can choose to prepare a custom model or use a pre-trained model from an online database e.g., TensorFlow Hub, Torchvision models, Hugging Face, etc. Furthermore, it is also expected that the model has been converted into the ONNX format. 

The image below outlines the workflow that is adopted when working with TensorRT.

![TensorRT Workflow](<../../../output/test/tracking_results/mobilevitv2_track/results/TensorRT Workflow.png>)

### 1) Build Phase

Import TensorRT Python API through the ```tensorrt``` module:
```
import tensorrt as trt
```
First, you need to create a logger which is included in the Python bindings that logs all messages preceding a certain severity to stdout.
```
logger = trt.Logger(trt.Logger.WARNING)
```
Alternatively, you can define your own implementation of the logger.
You then create a builder.
```
builder = trt.Builder(logger)
```
Building is an offline process and may take some time.

### 2) Create a Network Definition

The next step in optmizing a model is to create a network definition.

```
network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
```
The ``EXPLICIT_BATCH`` flag is required in order to import models using the ONNX parser. It basically specifies the batch dimension explicitly without leaving it to the builder to dynamically adjust.
This can be specified in the following manner:

```
builder.max_batch_size = 1
```

### 3) Importing a model using the ONNX Parser

The network definition created must be populated with the ONNX representation. This can be done using the ONNX parser in the following manner:

```
with open(onnx_file_path, "rb") as model:
    parser = trt.OnnxParser(network, logger)
    if not parser.parse(model.read()):
        raise RuntimeError("Failed to parse the ONNX file.")
```

### 4) Building an Engine

Perhaps the most important step is to create a build configuration specifying how TensorRT should optimize the model:

```
config = builder.create_builder_config()
```
This interface has many properties that you can set in order to control how TensorRT optimizes the network. One important property is the maximum workspace size. Layer implementations often require a temporary workspace, and this parameter limits the maximum size that any layer in the network can use. By default, the workspace is set to the total global memory size of the given device. However, we can limit it:
```
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 * 1024 * 1024 * 1024)  # 2GB
```
Additionally, it allows you to specify several important flags i.e., tensorrt.QuantizationFlag, tensorrt.DeviceType, tensorrt.ProfilingVerbosity, etc. We will look at the Quantization flag under the INT8 quantization section.

After the configuration has been specified, the engine can be built and serialized:
```
serialized_engine = builder.build_serialized_network(network, config)
```
It is useful to save the engine for future use, however, care must be taken since the engine created is specific to the hardware i.e., GPU model, and TensorRT verison, making it non-portable across platforms. 
```
with open(“sample.engine”, “wb”) as f:
    f.write(serialized_engine)
```
The resulting file needs to saved with ```.engine``` suffix.

### 5) Performing Inference

To perform inference you need to deserialize the engine using the Runtime interface. Like the builder, the runtime requires an instance of the logger:
```
runtime = trt.Runtime(logger)
```
You can then deserialize the engine from a memory buffer or load it from a file:
```
engine = runtime.deserialize_cuda_engine(serialized_engine)
                        OR
with open(“sample.engine”, “rb”) as f:
    serialized_engine = f.read()
```

The engine holds the optimized model, but to perform inference requires additional state for intermediate activations. This is done using the IExecutionContext interface:
```
context = engine.create_execution_context()
```

This is followed by allocating memory/buffers for the inputs and outputs. A sample scenario for a single input and output is as follows:
```
# Allocate CPU memory
output = np.empty([output_shape], dtype = target_dtype)

# Allocate device memory
d_input = cuda.mem_alloc(1 * input_batch.nbytes)
d_output = cuda.mem_alloc(1 * output.nbytes)

# Create bindings
bindings = [int(d_input), int(d_output)]

# Create CUDA stream
stream = cuda.Stream()
```
The bindings act as TRT pointers to the memory allocated. You then need to define a ```prediction function```,
that involves a copy from CPU RAM to GPU VRAM, executing the model, then copying the results back from GPU VRAM to CPU RAM:
```
def predict(batch):

    # transfer input data to device
    cuda.memcpy_htod_async(d_input, batch, stream)

    # execute model
    context.execute_async_v2(bindings, stream.handle, None)

    # transfer predictions back
    cuda.memcpy_dtoh_async(output, d_output, stream)

    # syncronize threads
    stream.synchronize()
    
    return output
```
To determine when inference (and asynchronous transfers) are complete, use the standard CUDA synchronization mechanisms. 

The purpose of asynchoronous execution/data transfer is to allow for better utilization of resources by enabling the CPU to perform other tasks while the GPU is actively processing or transferring data.


## Quantization

Model Quantization is a popular way of optimization which reduces the size of models thereby accelerating inference, while also opening up the possibilities of deployments on devices with lower computation power. Simply put, it is a process of mapping input values from a larger set to output values in a smaller set. In the context of deep learning, we often train deep learning models using floating-point 32 bit arithmetic (FP32) as we can take advantage of a wider range of numbers, resulting in more accurate models. The model parameters ``(weights and activations)`` are converted from this floating point representation to a lower precision representation, typically using 8-bit integers i.e., int8, range [-128, 127].

![Quantization Range Mapping](<../../../output/test/tracking_results/mobilevitv2_track/results/Quantization Range Mapping.jpeg>)

TensorRT supports the use of 8-bit integers to represent quantized floating point values. The quantization scheme is symmetric uniform quantization - quantized values are represented in signed INT8, and the transformation to quantized values uses the reciprocal scale, followed by rounding and clamping.

The quantization scheme depends on the chosen calibration algorithm to find a ``scale`` which best balances rounding error and precision error for specific data. There are two quantization workflows supported by TensorRT:

1) ``Post-Training Quantization (PTQ)``
It derives scale factors after the network has been trained. TensorRT provides a workflow for PTQ, called calibration, where it measures the distribution of activations within each activation tensor as the network executes on representative input data i.e., calibration data, then uses that distribution to estimate a scale value for the tensor.

2) ``Quantization-Aware Training (QAT)``
It computes scale factors during training. This allows the training process to compensate for the effects of the quantization and dequantization operations.

TensorRT’s [Quantization Toolkit](https://github.com/NVIDIA/TensorRT/tree/main/tools/pytorch-quantization) is a PyTorch library that can be used to perform optimization and supports both methods of quantization.

For the purpose of this task, I have worked with PTQ, and not QAT.

### TensorRT PTQ Workflow

![PTQ Workflow](<../../../output/test/tracking_results/mobilevitv2_track/results/PTQ Workflow.png>)


Lets discuss each step of the workflow one-by-one:

1) Trained Baseline Model

The baseline model is a trained model that has achieved acceptable accuracy. It can be in any format that is supported by TensorRT i.e., TensorFlow, PyTorch, etc. PTQ is effective and quick to implement since it does not require re-training the model.

2) Calibration

This step involves the use of a ``calibration dataset``, which in essence is a representative dataset taken form the testing or validation sets. To perform PTQ, we infer using the calibration set to determine the range of representable FP32 values to be quantized i.e., the scale that can be used to map the values to the quantized range .

    There are three popular techniques used to calibrate:
    - Min-Max: Use the minimum and maximum of the FP32 values seen during calibration.
    - Entropy: Minimize information loss between the original FP32 tensor and quantized tensor.
    - Percentile: Use the percentile of the distribution of absolute values seen during calibration.

3) Quantization

This step is where the actual quantization takes place. For this purpose use a quantization toolkit that is built for training and evaluating quantized models. We need to ensure that instead of the standard (original) modules we call quantized modules instead. An example of this is using the ``PyTorch Quantization Toolkit``:
```
quant_modules.initialize() # Calls qauntized modules instead of the original modules
```
For example, instead of ``Conv2d``, this wraps a qunatizer node around the inputs and outputs ``QuantConv2d``.

4) Export to ONNX (Optional)

Once We have calibrated and quantized the model using a toolkit of our choosing, we need to export the model to the ONNX format. This process serializes the model and enables the cross-platform deployability. ONNX format is an intermediate representation that supports all major machine learning frameworks and runtimes.

However, some toolkits provide the utility of calibrating the model using the ONNX format, thus not requiring any conversions, and can be directly converted to the optimized INT8 TensorRT engine. I have followed this path.

5) TensorRT Engine

Now we need to build an optimized TensorRT engine using the pipeline previously discussed, however with some minor additions to enable ``INT8`` precision.

Set the ``IBuilderConfig BuilderFlag`` to enable INT8 precison when building the engine. This allows for the selection of layers that INT8 precision.
```
# Set precision to INT8
config.set_flag(trt.BuilderFlag.INT8)
```
Additionally, set the ``IBuilderConfig QuantizationFlag`` to enable quantization, if not already performed. This enables quantization when building the engine, but you need to provide either an application-implemented or custom interface for calibration that enables the builder to choose a suitable scale factor for 8-bit inference. 

In addition, you need to specify the algorithm that performs this quantization. The most popular is the entropy based calibrator. This step can be performed in the following way:
```
# Enable INT8calibration
config.set_flag(trt.QuantizationFlag.IInt8EntropyCalibrator2)
```
6) Inference

Once the engine is built successfully with the right configuration and optimization profile, one simply needs to perform inference based on their inference pipleine using the deserialized engine. Conduct post-processing of generated outputs/results.

### Polygraphy

The default PTQ pipeline remains the same regardless of the framework being employed for the sake of calibration and subsequent quantization e.g, pytorch-quantization, neural network compression fraemwork, etc.
However, our codebase is quite complex, involving the use of multiple modules with a great number of intermediate layers. Furthermore, I was unable to install and import the ``pytorch-quantization`` toolkit due to version conflicts with the existing environment dependecies, thus I had to resort to another toolkit i.e., Polygraphy.

Polygraphy is a toolkit designed to assist in running and debugging deep learning models in various frameworks. It supports a number of backends for conducting inference such as Tensorflow, PyTorch, ONNX-Runtime, and TensorRT. The Python API can be found [here](https://docs.nvidia.com/deeplearning/tensorrt/polygraphy/docs/index.html).

You can install it either from source or using prebuilt wheels. Note: Polygraphy supports only Python 3.6 and later.

```
python -m pip install colored polygraphy --extra-index-url https://pypi.ngc.nvidia.com
```

Polygraphy Python API, encapsulates a lot of information from TensorRT, enabling ease of implementation but lesser control compared to the TensorRT Python API. It follows the same pattern for building the TensorRT engine. 

I have listed the steps below:

1) Calibration

    Define a ``Data Loader`` that takes in random or model inputs with the same shape and datatype, and return them as a dictionary for each model input. For instance:

    ```
    def calib_data():
    for _ in range(4):
        yield {"z": np.random.randn(1, 3, 128, 128).astype=np.float32,
            "x": np.random.randn(1, 3, 256, 256).astype=np.float32} 
    ```

    This is then fed to the ``Calibrator`` method supplied by the TensorRT backend. It supplies calibration data to TensorRT to calibrate the network for INT8 inference.

    ```
    calibrator = Calibrator(data_loader=calib_data(), cache="model-calib.cache")
    ```

    You can save the calibration output to a ``cache`` for reuseability.

2) Build TensorRT Engine

    Next, you build the ``network`` by first parsing the ONNX file and then creating the ``engine``, providing the ``calibrator`` created earlier.

    ```
    # Parse ONNX file
    builder, network, parser = network_from_onnx_path("model.onnx")

    # Create a TensorRT IBuilderConfig so that we can build the engine with INT8 enabled.
    config = create_config(builder, network, INT8=True, calibrator = calibrator)

    # Build Optimized INT8 TensorRT Engine
    with builder, network, parser, config:
        engine = engine_from_network((builder, network), config)
    ```

    This step creates an engine (deserialized) to be used directly. 
    
    Note: TensorRT Engine building excercise is time and resource intensive, in addition to being software and hardware (GPU) dependent. It is strongly advised to create a new engine for each inference session.

3) Saving the Engine

    However, it is vital that one saves the engine if the need arises (inferencing on the same platform, with the same virtual environment), to avoid wasting resouces. Since, our codebase inferences on a number of sequences, the optimal path is to save the engine and load it for inferencing.

    ```
    # To reuse the engine, we can serialize it and save it to a file.
    save_engine(engine, path=engine_file_path)
    ```

4) Inferencing

    The final step in the pipline. Simply, create a ``runner`` object using the engine created earlier. This way we ensure that the resouces are freed once inferencing is complete.

    ```
    with engine, TrtRunner(engine) as runner:
        # Provide input data
        z_input = np.random.randn(1, 3, 128, 128).astype=np.float32
        x_input = np.random.randn(1, 3, 128, 128).astype=np.float32


        # Infer
        outputs = runner.infer(feed_dict={"z": z_input, "x": x_input})
    ```

    Perform any post-processing based on the model and application use-case.

## Model Evaluation

The exisiting codebase provides functionality for processing model outputs for generating evaluation metrics i.e., bounding box predictions as well as inference time for each frame in a sequence. For the scope of this project, we are interested in viewing model inference speeds for all model formats.

### GOT-10k: Generic Object Tracking Benchmark

I have evaluated the model using ``GOT-10k: Generic Object Tracking Benchmark``, which is a large, high-diversity, one-shot database for generic object tracking in the wild. It consists of a number of standardized tests used to assess the performance, capabilities, or efficiency of hardware and/or software. 

Subsequently, I have evaluated the model using ``GOT-10K_TEST`` dataset which contains 180 sequences, each with a variable number of frames. The open-source dataset is available [here](http://got-10k.aitestunion.com/downloads). Model outputs generated via inference need to be uploaded onto to the [benchmark website](http://got-10k.aitestunion.com/submit_instructions) as per the specified instructions. To restructure and format these raw output files, go to the sub-driectory ``SMAT/misc`` and execute the following python script:

```
python file_structure.py
```

I have performed benchmarking for the following formats i.e., TensorRT FP32, TensorRT INT8, etc. 

Table 1 highlights metrics reported for each format.

![Benchmark Table](<../../../output/test/tracking_results/mobilevitv2_track/results/TensorRT Benchmark Metrics Table.png>)

Standardization and repeatability of the benchmark enables consistent results. Based on the reported metrics, there is a small speed-up of approximately 1.08x when inferencing with INT8 quantized TensorRT engine compared to the default FP32 precision engine. However, we observe a throughput of 24.82 fps using only TensorRT for engine building.

To visually represent the differences between each test case, please refer to the plot below:

![Benchmark Plot](<../../../output/test/tracking_results/mobilevitv2_track/results/TensorRT Benchmark Metrics Plot.png>)

### Analysis

There are several reasons for the insignificant speed-up with quantization and with TensorRT in general. Based on my observations and understanding I have listed some of them below:
- TensorRT is designed to optimize deep learning inference on NVIDIA CUDA platforms using techniques such as quantization, layer fusion, kernel tuning, etc. All of these have been employed, however, the codebase is extremely complex involving a large number of unique layers intricately stacked together. The complexity of the model poses a great challenge when generating an optimized TensorRT engine.
- The codebase has a strict adherence to the dependencies and their versions, which causes significant non-compatibility issues with TensorRT 8.5.3 and the CUDA toolkit. 
- Moreover, the underlying GPU architecture posed issues, as the usage of GPU memory and its release were inefficient. I have tried to implement a number of memory allocation procedures during inference, however, none yielded exceptional results.
- Another factor is the batch size. This is well documented that TensorRT works well large batch-sizes, however, SMAT inherently works with each individual frame of a sequence one-by-one i.e., batch-size 1, which means that variable batch sizes can not not be leveraged.
- Additionally, the serialized engine needs to be loaded from file and then deserialized before the ``predict`` method is called for each individual frame. This adds a large computational overhead whilst inferencing, diminishing inference speed.
- The TensorRT engine is not fully optimized. All weights have INT64 data types which require casting to INT32. Furthermore, there are multiple custom layers which are not fully optimized nor quantized. These layers could not be removed, impeding serialization of the network.
- Although, asynchronous excution was utilized when performing inference, however, the throughput does not seem to increase much. This may have deeper connections with the GPU architecture and its inability to fully optimize the inferencing pipeline for our usecase.
- Polygraphy is a powerful toolkit, however, it does not provide the granular control when building optimized TensorRT engines as that of TensorRT Python API. The slight increase in throughput for FP32 precision using TensorRT Python API is evident, since it allows the user to control certain aspects on engine building, resulting in faster inferencing.
- Although I was unable to successfully create a TensorRT quantized engine for INT8 precison using the TensorRT Python API, but based on the results achieved, I am certain the speed-up would not be significant (along the lines of the Polygraphy speed-up), due to the aforementioned reasons.
- Additionally, I have also observed a proportional relationship between the calibration data size and the accuracy of the model. When calibrated with a large number of samples, the INT8 quantized model performed better. 
- Furthermore, the memory foot-print of the INT8 engine is far less than that of FP32. However, based on the accuracy-speed tradeoff that was expected, the results are not very promising. 

To conclude, I believe TensorRT does not provide substantial performance gains with SMAT, however, one might be able to yield better results when only working with specific subgraphs or layers where it provides benefits, and falling back to native PyTorch where it does not.
