# Integrating Tflite into mediapipe
## Mediapipe Graph
A mediapipe graph represents the pipeline you will use to process the data. A Mediapipe graph consists of a series of nodes, each of which performs a specific operation on the input data. You can use the Mediapipe calculator graph tool to build your graph visually or you can write the code manually in C++ or Python.

Step 1: Create Custom Graph Directory
To create a custom solution, you would first need to create a folder in the medipaipe/mediapipe/graphs. To make things easier, just copy and edit a built-in solution graph made by mediapipe. But please note that you will need to edit it according to your needs. The bottomline is that you will need a BUILD file where you will list all of the calculators and dependencies that you need. Furthermore, you will need a pbtxt file containing the graph's nodes.

Step 2: Define Input and Output Streams
The next step is to define the input and output streams for your graph. Input streams provide data to the graph, while output streams capture the results of the processing. You can use the MediaPipe graph editor tool to define these streams or you can write the code manually.
```
input_stream: "input_video"
output_stream: "output_video"
```

Step 3: Add Nodes
Once you have your graph, you can start to add nodes that will perform the operations you need. Mediapipe provides a library of pre-built nodes for common operations like image manipulation, feature extraction, and classification. You can also create custom nodes using C++ or Python.
```
node {
    calculator: "FlowLimiterCalculator"
    input_stream: "input_video"
    input_stream: "FINISHED:output_video"
    input_stream_info: { 
      tag_index: "FINISHED",
      back_edge: true 
    }
    output_stream: "throttled_input_video"
}
```

Step 4: Connect Nodes
Once you have added your nodes, you need to connect them in the graph so that the output of one node becomes the input to another. You can use the MediaPipe graph editor tool to connect nodes visually or you can write the code manually.
```
node {
    calculator: "FlowLimiterCalculator"
    input_stream: "input_video"
    input_stream: "FINISHED:output_video"
    input_stream_info: { 
      tag_index: "FINISHED",
      back_edge: true 
    }
    output_stream: "throttled_input_video"
}

node: {

  calculator: "ImageTransformationCalculator"

  input_stream: "IMAGE:throttled_input_video"

  output_stream: "IMAGE:output_video"

  node_options: {

    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {

      output_width: 224

      output_height: 224

    }

  }

}
```

Step 5: Configure the Graph
Once you have defined your input and output streams, you need to configure the graph to specify how the data will be processed. This includes specifying things like the input image size, the output format, and the model parameters. You can use the MediaPipe graph editor tool to configure your graph or you can write the code manually.

A sample code snippet for modfying the input and output width of the image:
```
node: {

  calculator: "ImageTransformationCalculator"

  input_stream: "IMAGE:throttled_input_video"

  output_stream: "IMAGE:output_video"

  node_options: {

    [type.googleapis.com/mediapipe.ImageTransformationCalculatorOptions] {

      output_width: 224

      output_height: 224
    }
  }
}
```

Here's a second sample snippet. This is for using TfLiteInferenceCalculator to do predictions:
```
node {

  calculator: "TfLiteInferenceCalculator"

  input_stream: "TENSORS:image_tensor"

  output_stream: "TENSORS:output_tensor"

  node_options: {

    [type.googleapis.com/mediapipe.TfLiteInferenceCalculatorOptions] {

      model_path: "mediapipe/models/mnv3_large_224_v2_genderFalse-2.tflite"
    }
  }
}
```

Step 6: Run the Graph
Once you have configured your graph, you can run it on your input data. You can use the MediaPipe framework to process data in real-time or you can process batches of data offline. You can also use the MediaPipe debugger tool to visualize the output of each node in the graph.

Step 7: Optimize the Graph
Finally, you can optimize your graph to improve performance. This could include things like reducing the number of nodes in the graph, simplifying the model architecture, or using more efficient data formats. You can use the MediaPipe profiler tool to identify performance bottlenecks and optimize your graph accordingly.

In conclusion, creating a custom Mediapipe solution requires a clear understanding of the problem you want to solve, the ability to build a graph that represents the pipeline you will use to process the data, and the knowledge to optimize your graph for performance. With these steps, you can create powerful and efficient solutions for a wide range of computer vision and media processing tasks.

## Building Custom Solution (Windows)
Once you have created a graph for your custom mediapipe solution, you can build it using bazel.

Here's an example code snippet for building the desktop version:
```
bazel build -c opt --define MEDIAPIPE_DISABLE_GPU=1 mediapipe/examples/desktop/lips_segmentation:lips_segmentation_cpu
```

You will just need to edit the path so that it will suit your needs. 

This is a command for building the Lips Segmentation example in Mediapipe's Desktop Examples using Bazel build system. Here is the explanation of each part of the command:

- bazel is a build system that automates the building and testing of software. It uses a build file called BUILD to define the dependencies and build steps for a software project.
- build -c opt specifies the build configuration. In this case, we are building with the "opt" configuration, which enables all optimizations for the resulting binary.
- --define MEDIAPIPE_DISABLE_GPU=1 sets a Bazel build flag to disable GPU support for the Lips Segmentation example. This means that the example will only use CPU for processing and will not use any GPU resources.
- mediapipe/examples/desktop/lips_segmentation:lips_segmentation_cpu specifies the target to build, which is the Lips Segmentation example for the CPU.
- mediapipe/examples/desktop/lips_segmentation is the directory path for the Lips Segmentation example, and lips_segmentation_cpu is the name of the target binary for the CPU version.
- Overall, this command builds the Lips Segmentation example using Bazel build system with optimized build configuration, disabling GPU support and targeting the CPU version of the example.

## Running your Custom Solution (Windows)
To run your custom solution, you should have first finished the building of your custom mediapipe solution. Once you are done, you can proceed to this step.

Here's an example code snippet for running your solution with your own pbtxt file:
```
bazel-bin\mediapipe\examples\desktop\lips_segmentation\lips_segmentation_cpu --calculator_graph_config_file=mediapipe/graphs/lips_segmentation/lips_segmentation_desktop_live.pbtxt
```

This is a command that runs the Lips Segmentation example in Mediapipe's Desktop Examples using the CPU version of the Lips Segmentation binary that was built with Bazel. Here is the explanation of each part of the command:

- bazel-bin\mediapipe\examples\desktop\lips_segmentation\lips_segmentation_cpu is the path to the binary file that was built with Bazel. This binary is the CPU version of the Lips Segmentation example, which means it does not use any GPU resources for processing.
- --calculator_graph_config_file=mediapipe/graphs/lips_segmentation/lips_segmentation_desktop_live.pbtxt specifies the path to the configuration file for the Lips Segmentation graph. This configuration file is in Protocol Buffer Text format and defines the structure and parameters of the Lips Segmentation graph.
- When this command is executed, the Lips Segmentation example will run using the CPU version of the binary and the configuration file specified. The example will process live video from the default webcam and display the output in a window on the screen. The Lips Segmentation graph will detect and segment the lips in the video, and color them in red.

# Creating Custom Mediapipe Calculator
## General Guide
- Install Mediapipe: To get started with Mediapipe, you'll need to have it installed on your system. You can install Mediapipe by following the instructions in the official documentation.

- Define the input and output streams: Before creating the calculator, you need to define the input and output streams that it will use. Input streams are the data that the calculator receives from other parts of the pipeline, while output streams are the data that it sends to other parts of the pipeline. You can define these streams in a separate file called a .pbtxt file.

- Create the calculator: To create the calculator, you'll need to write a new C++ class that inherits from the mediapipe::CalculatorBase class. This class provides the basic functionality that all calculators require, such as input and output stream management, as well as access to the calculator context.

- Define the calculator parameters: Once you've created the calculator class, you can define the parameters that it will use. Parameters are variables that can be set from the pipeline configuration file, and allow you to adjust the behavior of the calculator without changing its source code.

- Implement the calculator's processing function: The most important part of the calculator is its processing function, which takes in input data, processes it in some way, and outputs the results. This function should be implemented in your calculator class.

- Build and run the calculator: Once you've written your calculator code, you can build it using the CMake build system that comes with Mediapipe. After building your calculator, you can test it by running it in a Mediapipe graph.

- Debug and refine the calculator: Finally, you can debug and refine your calculator by examining its input and output streams, as well as its internal state, using the Mediapipe visualizer tool.

## Creating calculator file
First step is to create a calculator file from whichever folder in the mediapipe/mediapipe/calculators

However, in this example, I will use the directory mediapipe/mediapipe/calculators/tflite and created a file
```
tflite_print_output_tensor_calculator.cc
```

## Edit directory Build File
Since I have decided to place my custom calculator in the tflite folder, I will need to edit the Build file located in mediapipe/mediapipe/calculators/tflite.

This is so that our custom calculator will be recognized and built.

Here's a sample code snippet that you will need to add in the build file:
```
cc_library(
    name = "tflite_print_output_tensor_calculator",
    srcs = ["tflite_print_output_tensor_calculator.cc"],
    visibility = ["//visibility:public"],
    deps = ["//mediapipe/framework:calculator_framework",
            "//mediapipe/framework/formats:tensor",
            "//mediapipe/framework/port:logging",
            "//mediapipe/framework/port:ret_check",
            "@org_tensorflow//tensorflow/lite:framework",
            ],
    alwayslink = 1,
)
```

This rule defines a C++ library target named tflite_print_output_tensor_calculator that depends on several other libraries and source files.

Here is a breakdown of the various arguments that are passed to the cc_library rule:

- name: This specifies the name of the library target. In this case, the name is tflite_print_output_tensor_calculator.

- srcs: This specifies the source files that are used to build the library target. In this case, the source file tflite_print_output_tensor_calculator.cc is used.

- visibility: This specifies the visibility of the library target. In this case, the target is marked as public, which means it can be used by other targets outside of this package.

- deps: This specifies the dependencies of the library target. There are several dependencies in this case, including the calculator_framework and tensor libraries from the Mediapipe framework, as well as the logging and ret_check libraries from the Mediapipe port library. Additionally, there is a dependency on the tensorflow/lite:framework library from the Tensorflow Lite library.

- alwayslink: This specifies that the library should always be linked into the binary, even if it's not directly used by any other targets.

Overall, this cc_library rule is used to build a C++ library that provides a calculator for printing the output tensor of a Tensorflow Lite model in a Mediapipe graph.

## Edit graph Build File
To use the custom calculator on a specific, you will also need to edit the specific build file in your solutions directory.

Here's a sample code snippet, edit it to suit your needs:
```
cc_library(
    name = "desktop_tflite_calculators",
    deps = [
        "//mediapipe/calculators/core:concatenate_vector_calculator",
        "//mediapipe/calculators/core:flow_limiter_calculator",
        "//mediapipe/calculators/core:previous_loopback_calculator",
        "//mediapipe/calculators/core:split_vector_calculator",
        "//mediapipe/calculators/image:image_transformation_calculator",
        "//mediapipe/calculators/tflite:tflite_print_output_tensor_calculator", # Added this
        "//mediapipe/calculators/tflite:ssd_anchors_calculator",
        "//mediapipe/calculators/tflite:tflite_converter_calculator",
        "//mediapipe/calculators/tflite:tflite_inference_calculator",
        "//mediapipe/calculators/tflite:tflite_tensors_to_detections_calculator",
        "//mediapipe/calculators/util:annotation_overlay_calculator",
        "//mediapipe/calculators/util:detection_label_id_to_text_calculator",
        "//mediapipe/calculators/util:detections_to_render_data_calculator",
        "//mediapipe/calculators/util:non_max_suppression_calculator",
        "//mediapipe/calculators/video:opencv_video_decoder_calculator",
        "//mediapipe/calculators/video:opencv_video_encoder_calculator",
        "//mediapipe/framework:calculator_framework",
    ],
)
```

# Custom Mediapipe Calculator Breakdown
## Calculator Breakdown: Includes
This contains several include statements that import header files from various libraries used in a Mediapipe project.

Here's a sample code snippet. You will most likely need other headers as include:
```
#include "mediapipe/framework/calculator_framework.h"

#include "tensorflow/lite/model.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/input_stream.h"

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/logging.h"
```

Explanation for each include:

- #include "mediapipe/framework/calculator_framework.h": This includes the header file for the Mediapipe calculator framework, which provides the building blocks for creating custom calculators in Mediapipe.

- #include "tensorflow/lite/model.h": This includes the header file for the Tensorflow Lite model library, which is used for loading and running Tensorflow Lite models in a Mediapipe project.

- #include "mediapipe/framework/packet.h": This includes the header file for the Mediapipe packet class, which is used to pass data between calculators in a Mediapipe graph.

- #include "mediapipe/framework/packet_type.h": This includes the header file for the Mediapipe packet type library, which provides functionality for handling different types of data packets in a Mediapipe graph.

- #include "mediapipe/framework/input_stream.h": This includes the header file for the Mediapipe input stream class, which provides functionality for reading input packets from a Mediapipe graph.

- #include "mediapipe/framework/formats/tensor.h": This includes the header file for the Mediapipe tensor format library, which provides functionality for working with tensors (multi-dimensional arrays) in a Mediapipe project.

- #include "mediapipe/framework/port/logging.h": This includes the header file for the Mediapipe logging library, which provides logging functionality for debugging and error reporting in a Mediapipe project.

Together, these header files provide the necessary functionality and data structures for creating custom calculators in Mediapipe that can load and run Tensorflow Lite models, work with tensors, and pass data between calculators in a graph.

## Calculator Breakdown: Declaring Global Variables
The code snippet you provided declares a constexpr array of characters named kTensorsTag with the value "TENSORS".

```
constexpr char kTensorsTag[] = "TENSORS";
```

In C++, a constexpr variable is a variable whose value is known at compile time and cannot be changed at runtime. By using constexpr, the compiler can evaluate the expression and substitute its value wherever it is used in the code.

In this case, kTensorsTag is a constexpr array of characters that contains the string "TENSORS". The [] operator is used to access individual characters in the array.

This string is used as a tag to identify a stream of data packets in a Mediapipe graph that contains tensors (multi-dimensional arrays of data). By using a constant string like "TENSORS" as the tag, it ensures that the same tag is used consistently throughout the code and simplifies the process of referring to this stream of data packets.

## Calculator Breakdown: Custom Calculator Class
This code snippet defines a custom calculator named TflitePrintOutputTensorCalculator in a Mediapipe project. The purpose of this calculator is to print the output tensor values to the console.

```
/ A custom calculator that prints the output tensor values to the console.
class TflitePrintOutputTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(TflitePrintOutputTensorCalculator);
```

The class definition begins with the keyword class followed by the name of the class (TflitePrintOutputTensorCalculator). The class is derived from the CalculatorBase class provided by the Mediapipe framework, which defines the basic functionality for a calculator.

The public section of the class definition includes the following member functions:

- static absl::Status GetContract(CalculatorContract* cc);: This static function is used to define the input and output streams that the calculator expects and produces, respectively. It takes a CalculatorContract object as an argument, which is used to define the streams.

- absl::Status Open(CalculatorContext* cc) override;: This function is called when the calculator is opened, and can be used to initialize any resources needed by the calculator.

- absl::Status Process(CalculatorContext* cc) override;: This function is called whenever new input data is available for the calculator to process. It performs the actual computation and produces output data.

The REGISTER_CALCULATOR macro at the end of the class definition is used to register the calculator with the Mediapipe framework. This makes the calculator available to be used in a Mediapipe graph.

By defining a custom calculator, you can extend the functionality of the Mediapipe framework to perform specific tasks that are not provided by the built-in calculators. In this case, the TflitePrintOutputTensorCalculator is used to print the output tensor values to the console.

## Calculator Breakdown: GetContract
This code snippet shows the implementation of the GetContract function for the TflitePrintOutputTensorCalculator class in a Mediapipe project. This function is called when the calculator is initialized to define the input and output streams that the calculator expects and produces, respectively.

```
// GetContract
absl::Status TflitePrintOutputTensorCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  return absl::OkStatus();
}
```

The GetContract function takes a CalculatorContract object as an argument, which is used to define the input and output streams. In this case, the function defines a single input stream with the tag kTensorsTag, which is the constant string "TENSORS" that identifies a stream of data packets in a Mediapipe graph that contains tensors. The Set method is called to specify the data type of the input stream as a vector of TfLiteTensor objects, which is a type defined in the TensorFlow Lite library.

The function returns an absl::Status object with the status of the operation. In this case, the function always returns an OK status, indicating that the input stream was successfully defined.

By defining the input and output streams in the GetContract function, the Mediapipe framework can verify that the inputs and outputs of the calculator are compatible with other calculators in the graph and can allocate the necessary resources for processing the data.


## Calculator Breakdown: Open
This code snippet shows the implementation of the Open function for the TflitePrintOutputTensorCalculator class in a Mediapipe project. This function is called when the calculator is opened, and can be used to initialize any resources needed by the calculator.

```
// Open
absl::Status TflitePrintOutputTensorCalculator::Open(CalculatorContext* cc) {
  return absl::OkStatus();
}
```

In this particular case, the Open function does not perform any initialization and simply returns an absl::OkStatus object, indicating that the calculator was successfully opened.

If the calculator requires initialization, the Open function can be used to perform any necessary setup, such as allocating memory, initializing variables, or setting up connections to external devices or services. Any resources that are allocated or initialized in the Open function should be released or cleaned up in the Close function, which is called when the calculator is closed.

By defining the Open function, you can customize the behavior of the calculator when it is initialized and make sure that any necessary resources are properly set up before processing data.


## Calculator Breakdown: Process
This code snippet shows the implementation of the Process function for the TflitePrintOutputTensorCalculator class in a Mediapipe project. This function is called for each input packet that is received by the calculator, and is used to perform the actual processing of the data.

```
// Process
absl::Status TflitePrintOutputTensorCalculator::Process(CalculatorContext* cc) {
  RET_CHECK(!cc->Inputs().Tag(kTensorsTag).IsEmpty());
  const auto& input_tensors = cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
  
  // Loop through all the output tensors and print their values.
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& output_tensor = input_tensors[i];
    const float* output_tensor_flat = output_tensor.data.f;

    const int num_elements = output_tensor.bytes / sizeof(float);
    // Print the values in the output tensor to the console.
    for (int j = 0; j < num_elements; ++j) {
      LOG(ERROR) << "Output tensor " << i << " value at index " << j << ": " << output_tensor_flat[j];
    }
  }

  return absl::OkStatus();
}
```

In this particular case, the Process function retrieves the input tensors from the cc object, which contains the input streams for the calculator. It first checks that the input stream with tag kTensorsTag is not empty using the IsEmpty function. If the input stream is empty, it returns an error status using the RET_CHECK macro.

Assuming that the input stream is not empty, the function loops through all the output tensors and prints their values to the console. For each tensor, it first retrieves the raw data buffer using the data member of the TfLiteTensor struct. In this case, the data is assumed to be a flat array of float values, so it is cast to a const float* pointer.

The number of elements in the tensor is calculated by dividing the number of bytes in the tensor by the size of a float using the bytes member of the TfLiteTensor struct. The function then loops through all the elements in the tensor and prints their values to the console using the LOG macro from the Mediapipe framework.

Finally, the function returns an absl::OkStatus object, indicating that the processing was completed successfully.

By defining the Process function, you can customize the behavior of the calculator when processing data and perform any necessary computations or transformations on the input data.


## Entire Code
Here's the entire code for this custom calculator:
```
#include "mediapipe/framework/calculator_framework.h"

#include "tensorflow/lite/model.h"
#include "mediapipe/framework/packet.h"
#include "mediapipe/framework/packet_type.h"
#include "mediapipe/framework/input_stream.h"

#include "mediapipe/framework/formats/tensor.h"
#include "mediapipe/framework/port/logging.h"

namespace mediapipe {

constexpr char kTensorsTag[] = "TENSORS";

// A custom calculator that prints the output tensor values to the console.
class TflitePrintOutputTensorCalculator : public CalculatorBase {
 public:
  static absl::Status GetContract(CalculatorContract* cc);

  absl::Status Open(CalculatorContext* cc) override;

  absl::Status Process(CalculatorContext* cc) override;
};
REGISTER_CALCULATOR(TflitePrintOutputTensorCalculator);

// GetContract
absl::Status TflitePrintOutputTensorCalculator::GetContract(CalculatorContract* cc) {
  cc->Inputs().Tag(kTensorsTag).Set<std::vector<TfLiteTensor>>();
  return absl::OkStatus();
}

// Open
absl::Status TflitePrintOutputTensorCalculator::Open(CalculatorContext* cc) {
  return absl::OkStatus();
}

// Process
absl::Status TflitePrintOutputTensorCalculator::Process(CalculatorContext* cc) {
  RET_CHECK(!cc->Inputs().Tag(kTensorsTag).IsEmpty());
  const auto& input_tensors = cc->Inputs().Tag(kTensorsTag).Get<std::vector<TfLiteTensor>>();
  
  // Loop through all the output tensors and print their values.
  for (int i = 0; i < input_tensors.size(); ++i) {
    const auto& output_tensor = input_tensors[i];
    const float* output_tensor_flat = output_tensor.data.f;

    const int num_elements = output_tensor.bytes / sizeof(float);
    // Print the values in the output tensor to the console.
    for (int j = 0; j < num_elements; ++j) {
      LOG(ERROR) << "Output tensor " << i << " value at index " << j << ": " << output_tensor_flat[j];
    }
  }

  return absl::OkStatus();
}

}  // namespace mediapipe

```
