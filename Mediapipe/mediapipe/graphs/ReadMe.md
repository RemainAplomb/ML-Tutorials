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