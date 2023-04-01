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

2. To create a dataset for Deeplab, you can refer to this SOP: [https://gitlab.id-yours.com/rahmani/lips-segmentation/-/blob/main/Generate-Tfrecord.pdf](https://github.com/RemainAplomb/ML-Tutorials/blob/main/Tensorflow/Deeplab/Generate-Tfrecord.pdf)

3. To train a model using Tensorflow's deeplab, you can refer to this SOP: [https://gitlab.id-yours.com/rahmani/lips-segmentation/-/blob/main/DeeplabTraining_SOP.pdf](https://github.com/RemainAplomb/ML-Tutorials/blob/main/Tensorflow/Deeplab/DeeplabTraining_SOP.pdf)

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
