# ML-Tutorials
This will contain Machine Learning related tutorials/guides that I made

# Tensorflow
## Brief Introduction
TensorFlow is an open-source software library for data flow and differentiable programming across a range of tasks. It is a symbolic math library, and is also used for machine learning applications such as neural networks. TensorFlow was developed by the Google Brain team and is used in many of Google's products and services. It is designed to be flexible, efficient and portable, and it allows for the deployment of computation to one or more CPUs or GPUs in a desktop, server, or mobile device with a single API. TensorFlow also provides a rich set of tools for debugging and visualization.

## Basic Resources
### Tensorflow Repository

Visit the github repository located at : https://github.com/tensorflow/tensorflow

To work with it, you must clone it by running git on your desired directory:
!git clone https://github.com/tensorflow/models.git


# Deeplab
## Brief Introduction
DeepLab is a state-of-the-art deep learning model for semantic image segmentation. It was developed by Google Research team, and it uses a convolutional neural network (CNN) to produce a dense per-pixel prediction of the object class labels in an image. The model is built on top of TensorFlow, and it is trained on a large dataset of images and their corresponding annotations.

DeepLab models are pre-trained on the COCO-stuff and PASCAL VOC datasets and fine-tuned on specific datasets of interest. The models offer a range of architectures, including MobileNet-v2 and Xception, that enable efficient computation on both high-end and low-end devices. The DeepLab models are widely used in many computer vision applications such as object detection, image captioning, and video segmentation.

## Basic Resources
1. To see Google's deeplab, you can visit the tensorflow's library. Specifically, in "models/research/deeplab"

2. To create a dataset for Deeplab, you can refer to this SOP: https://gitlab.id-yours.com/rahmani/lips-segmentation/-/blob/main/Generate-Tfrecord.pdf

3. To train a model using Tensorflow's deeplab, you can refer to this SOP: https://gitlab.id-yours.com/rahmani/lips-segmentation/-/blob/main/DeeplabTraining_SOP.pdf

## Create a Dataset for Deeplab
Creating a dataset for use with DeepLab can be a multi-step process. Here is a general outline of how to create a dataset for semantic image segmentation using DeepLab:

Collect a set of images: Gather a set of images that are representative of the task you want to perform, such as object detection or image captioning.

Annotate the images: Annotate the images by drawing bounding boxes around the objects of interest and labeling them with their corresponding class or object names.

Convert the annotations to a format that can be used by DeepLab: The annotations can be converted to the Pascal VOC or COCO annotation format, which is a JSON file that contains the bounding boxes, class labels, and other information about the objects in the image.

Split the dataset into train, validation, and test sets: Divide the dataset into three sets, one for training, one for validation, and one for testing.

Preprocess the images: Resize or crop the images to the desired size, normalize the pixel values, and apply any other preprocessing steps as needed.

Create TFRecord files: Convert the images and annotations into a format that can be used by TensorFlow by creating TFRecord files, which are a binary file format used to store tensors.

Use the dataset to train a DeepLab model: Use the train and validation sets to train a DeepLab model, and use the test set to evaluate its performance.

It is important to note that this is a general approach, depending on the specific dataset and task, the process may differ. Additionally, there are also several tools and libraries available that help to simplify the annotation process and converting the annotations to the desired format.

## Train Model using Deeplab
Prepare the dataset: Convert the images and annotations into a format that can be used by TensorFlow by creating TFRecord files, which are a binary file format used to store tensors.

Choose a pre-trained DeepLab model: Select a pre-trained DeepLab model that is appropriate for your task, such as Mobilenetv2, Mobilenetv3 small, and Mobilenetv3 large.

Load the pre-trained model: Use TensorFlow's pretrained models. It can be found here https://github.com/tensorflow/models/blob/master/research/deeplab/g3doc/model_zoo.md

Define the optimizer: Choose an optimizer such as Adam or Momentum.

Define learning rate: Learning rate will affect how much the model will change when backpropagating. Larger learning rate could mean faster training, but it is susceptible to being stuck on local minimas. As such, you should experiment with this

Train the model: Use the train.py provided in models/research/deeplab/train.py

Validate the model: Use the eval.py provided in models/research/deeplab/eval.py

Visualize the model: Use the vis.py provided in models/research/deeplab/vis.py

Save the model: Export the pb file using  models/research/deeplab/export_model.py


# Creating Model Architectures
## Backbone
For the feature extractor of a model, you can either create your own feature extractor or use the backbones provided by Tensorflow.

An example of this is MobileNetV3Large, MobileNetV3Small, and MobileNetV2. To get these model's backbone, use this code snippet and edit it to suit your needs.


```c++
inputs = L.Input(input_shape)

backbone = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)

backbone.trainable = False

x = backbone.output
```

## Model's Head
The model's head will use the output of the feature extractor, and run it to different neural networks so that the model will be able to predict the groundtruth.

### Single Head
Typically, you will only need a single head for the model. Here's an example code snippet which only predicts the gender.

```
# Gender Detection
gender_dense1 = L.Dense(128, activation='relu', name='gender_dense1')(x)
gender_flatten = L.Flatten( name='gender_flatten' (gender_dense1)
gender_dense2 = L.Dense(128, activation='relu', name='gender_dense2')(gender_flatten)
gender_output = L.Dense(1, activation='sigmoid', name='gender_output')(gender_dense2)

model = Model(inputs=[inputs], outputs=[gender_output])
```
### Multiple Head/Branches
In the case that you need to create a tensorflow model that detects different kinds of things. You will have to make seperate heads/branches for each of your desired output. 

Here's an example code snippet:
```
# Face Landmark
landmark_branch = tf.keras.Sequential([L.GlobalAveragePooling2D(name='landmark_pooling'), L.Dropout(0.2, name='landmark_dropout'), L.Dense(num_landmarks*2, activation="sigmoid", name='landmark_output')], name='landmark_branch')
landmark_output = landmark_branch(x)

# Gender Detection
gender_branch = tf.keras.Sequential([
        L.Dense(128, activation='relu', name='gender_dense1'),
        L.Flatten(name='gender_flatten'),
        L.Dense(128, activation='relu', name='gender_dense2'),
        L.Dense(1, activation='sigmoid', name='gender_output')], name='gender_branch')
gender_output = gender_branch(x)

gender_branch.trainable = False
for layer in gender_branch.layers:
	layer.trainable = False

model = Model(inputs=[inputs], outputs=[gender_output, landmark_output])
```

There are cases where training multiple branch at the same time creates a conflict and results to the training being stuck at a local minima. To avoid this, we need to only train a single branch a time, and freeze the other branches.

## Entire Code
```
def build_model(input_shape, num_landmarks):
    inputs = L.Input(input_shape)

    backbone = MobileNetV3Large(include_top=False, weights="imagenet", input_tensor=inputs)
    backbone.trainable = False

    x = backbone.output

    # Face Landmark
    landmark_branch = tf.keras.Sequential([
        L.GlobalAveragePooling2D(name='landmark_pooling'),
        L.Dropout(0.2, name='landmark_dropout'),
        L.Dense(num_landmarks*2, activation="sigmoid", name='landmark_output')
    ], name='landmark_branch')
    landmark_output = landmark_branch(x)

    # Gender Detection
    gender_branch = tf.keras.Sequential([
        L.Dense(128, activation='relu', name='gender_dense1'),
        L.Flatten(name='gender_flatten'),
        L.Dense(128, activation='relu', name='gender_dense2'),
        L.Dense(1, activation='sigmoid', name='gender_output')
    ], name='gender_branch')
    gender_output = gender_branch(x)

    gender_branch.trainable = False
    for layer in gender_branch.layers:
        layer.trainable = False

    model = Model(inputs=[inputs], outputs=[gender_output, landmark_output])
    return model
```


# Loading/Processing Datasets
## Pandas
When training your model, one method to laod and process your dataset is to use Pandas. However, when you use this, the data will be loaded into the machine's RAM. As such, if you have a dataset which takes a lot of space, it will most likely crash your device as the RAM will be used up.

But still, if you insist on using this, here's a sample code snippet:
```
import pandas as pd

# convert to dataframe
df = pd.DataFrame()
df['image'], df['landmarks'], df['gender'] = train_x, train_y_landmarks, train_y_gender
#df['image'] = train_x
df.head()
```

And if you want to process your pandas dataframe, you can use something like this:
```
def read_image_lankmarks(landmark_paths):
    result_landmarks = []
    for image_landmark_path in tqdm(landmark_paths):
      """ Lankmarks """
      data = open(image_landmark_path, "r").read()
      lankmarks = []

      for line in data.strip().split("\n")[1:]:
          x, y = line.split(" ")
          x = float(x)/512
          y = float(y)/512

          lankmarks.append(x)
          lankmarks.append(y)

      lankmarks = np.array(lankmarks, dtype=np.float32)
      result_landmarks.append(lankmarks)
    result_landmarks = np.array(result_landmarks)
    return result_landmarks

y_landmarks = read_image_lankmarks(df['landmarks'])
```

## Tensorslice dataset (Recommended)
The best way that I currently know how to load and process a dataset is to use tensorflow's built-in functionalities for it. Basically, you will need to create multiple functions which will only be called and used by Tensorflow during training. This prevents the data from clogging up in the RAM.

Here's a sample code snippet for:
```
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def load_dataset(path):
    train_x = sorted(glob(os.path.join(path, "train", "images", "*.jpg")))
    train_y_landmarks = []
    train_y_gender = []
    for path in train_x:
      temp_split1 = path.split("/")
      temp_split2 = temp_split1[-1].split(".")
      temp_split3 = temp_split2[0].split("-")
      if temp_split3[-1] == "1" or temp_split3[-1] == "0":
        temp_gender = temp_split3[-1]
      else:
        temp_gender = "0"
        print("Error gender annotate in path: " + path + ", " + temp_split3[-1])
      
      temp_landmarks = "/content/LaPa/train/landmarks/" + temp_split3[0] + ".txt"
      train_y_landmarks.append(temp_landmarks)

      #temp_train_y = [train_y_landmarks, train_y_gender]
      train_y_gender.append(temp_gender)
    print( train_x[1], train_x[-1])
    print( train_y_landmarks[1], train_y_landmarks[-1])
    print( train_y_gender[1], train_y_gender[-1])

    return train_x, train_y_landmarks, train_y_gender

def read_image_lankmarks(landmark_path, gender):
    image = cv2.imread(landmark_path, cv2.IMREAD_GRAYSCALE)
    h, w = image.shape


    temp_split1 = landmark_path.split("/")
    temp_split2 = temp_split1[-1].split("-")
    temp_filename = "/content/LaPa/train/landmarks/" + temp_split2[0] + ".txt"

    temp_split3 = temp_split2[-1].split(".")
    temp_gender = float(temp_split3[0])

    """ Lankmarks """
    data = open(temp_filename, "r").read()
    lankmarks = []
    gender_list = [float(gender)]
    
    if temp_gender != gender_list[0]:
      print( "Error in: " + landmark_path)
      print( "Given gender: " + str(gender))
      print( "Gender found: " + temp_split3[0])

    for line in data.strip().split("\n")[1:]:
        x, y = line.split(" ")
        x = float(x)/w
        y = float(y)/h

        lankmarks.append(x)
        lankmarks.append(y)

    lankmarks = np.array(lankmarks, dtype=np.float32)
    gender_list = np.array(gender_list, dtype=np.float32)
    return lankmarks, gender_list

def read_image_preprocess(image_path):
    """ Image """
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    #h, w, _ = image.shape
    image = cv2.resize(image, (image_w, image_h))
    image = image/255.0
    image = image.astype(np.float32)

    return image

def preprocess1(x):
    def f(x):
        x = x.decode()
        image= read_image_preprocess(x)
        return image

    image_temp_list= tf.numpy_function(f, [x], [tf.float32])
    image = image_temp_list[0]
    image.set_shape([image_h, image_w, 3])
    return image

def preprocess2(y_genders, y_landmarks):
    def f(y_genders, y_landmarks):
        temp_y_landmarks = y_landmarks.decode()
        #print(y_genders)
        temp_y_gender = y_genders.decode()
        #print(y_landmarks)

        landmarks, y_gender = read_image_lankmarks(temp_y_landmarks, temp_y_gender)
        return y_gender, landmarks
    print( "x")
    gender, landmarks = tf.numpy_function(f, [y_genders, y_landmarks], [tf.float32, tf.float32])
    print( "xx")
    landmarks.set_shape([num_landmarks * 2])
    gender.set_shape([1])
    print(gender.shape)
    return gender, landmarks

def tf_dataset(x, y_genders, y_landmarks, batch=8):
    print("tf_dataset 1")
    ds1 = tf.data.Dataset.from_tensor_slices(x)
    ds1 = ds1.map(preprocess1)

    ds2 = tf.data.Dataset.from_tensor_slices((y_genders, y_landmarks))
    print("tf_dataset 2")
    ds2 = ds2.map(preprocess2)

    dataset = tf.data.Dataset.zip((ds1, ds2))
    dataset = dataset.batch(batch).prefetch(2)
    return dataset
```

To use it, you can follow this code snippet:

```
""" Paths """
dataset_path = "/content/LaPa"
model_path = "/content/drive/MyDrive/Gender-Landmark_224/Results/mnv3_large_224_v2_genderFalse-3/model.h5"
csv_path = "/content/drive/MyDrive/Gender-Landmark_224/Results/mnv3_large_224_v2_genderFalse-3/data.csv"

""" Loading the dataset """
#(train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_dataset(dataset_path)
train_x, train_y_landmarks, train_y_gender = load_dataset(dataset_path)
print(f"Train: {len(train_x)}/{len(train_y_landmarks)}/{len(train_y_gender)}")
```


# Compiling Model and Loading Checkpoints
## Compiling
To start training your model, you will first need to compile it. 

To do that, you can follow this code snippet:
```
""" Model """
model = build_model(input_shape, num_landmarks)
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=['binary_crossentropy', 'mean_squared_error'],
              loss_weights=[0, 1],
              metrics=['accuracy'])
```

Please note that, the training will be affected by what you choose as optimizer and other parameters. As for loss_weights, it will just determine the ratio in which each loss branch will affect the final loss of the model.

In this case, the loss for gender(i.e. binary_crossentropy) will not contribute to the final loss value since the loss weight for it is set to 0. I have set it to 0 because the gender branch in this has been frozen.

## Loading Checkpoints
To load a checkpoint, you will need a .ckpt file

Here's the code snippet for loading a .ckpt file as checkpoint:
```
""" Model """
model.load_weights( "/content/drive/MyDrive/Gender-Landmark_224/Checkpoints/mnv3_large_224_v2_genderFalse-2/Gender_e30.ckpt")
model.compile(optimizer=tf.keras.optimizers.Adam(lr),
              loss=['binary_crossentropy', 'mean_squared_error'],
              loss_weights=[0, 1],
              metrics=['accuracy'])
```

Please note that after loading the weights, you must compile/recompile the model.

# Training Model and Setting up Callbacks
## Set up Callbacks
To have a reliable training, you will need to set up callbacks that will be called upon after each epoch. An example of a callback is a function that saves a checkpoint of your model training. However, there are also other types of callbacks, it will depend on which one you would like to use and set up.

Here's a sample code snippet:
```
callbacks = [
    ModelCheckpoint(model_path, verbose=1, save_best_only=True, monitor='val_loss'),
    ModelCheckpoint(checkpoint_path, save_weights_only=True, verbose= 1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-10, verbose=1),
    CSVLogger(csv_path, append=True),
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False)
]
```

In the provided snippet above, we are using 5 functions that will be executed during callback. The first one will save a copy of the model in .h5 format. The second, it will save a model checkpoint in .ckpt format. The third is for dynamic changing of the learning rate. If the loss doesn't change within 5 epochs, the learning rate will be reduced by a factor of 0.1. We also have the CSVLogger, which as the name implies will log the training result data into a csv file. Lastly, a function for early stopping, this will be triggered if within 20 epochs there are no improvement in the training.

## Training/Fitting

Training is pretty simple to set up, you would just need to follow this code snippet and change it according to your needs:

```
history = model.fit(train_ds,
    validation_data=train_ds,
    epochs=100,
    callbacks=callbacks
)
```





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
