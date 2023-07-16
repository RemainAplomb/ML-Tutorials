# Image Segmentation

## Overview

Image segmentation is a vital process in computer vision. It involves partitioning a digital image into multiple segments, typically to locate objects and boundaries. In more specific terms, image segmentation's goal is to assign a label to every pixel in an image such that pixels with the same label share certain visual characteristics.

In the process of developing an effective image segmentation model, we have tried various strategies, including popular models like Deeplab, Segformer, and UNET, and even custom, modified versions of Vision Transformers.

## We have tried

### Deeplab

DeepLab is a state-of-art deep learning model for semantic image segmentation, where the goal is to assign semantic labels (e.g., dog, cat, car) to every pixel in the input image. DeepLab uses an atrous convolution (also known as a dilated convolution) to explicitly control the resolution at which feature responses are computed within Deep Convolutional Neural Networks. The model also includes use of atrous spatial pyramid pooling to robustly segment objects at multiple scales, and improved training procedures for tackling the challenging issues in semantic image segmentation.

Training:

```
!python3 deeplab/train.py \
--logtostderr \
--train_split="train" \
--model_variant="mobilenet_v3_small_seg" \
--train_crop_size="257,257" \
--train_batch_size=4 \
--dataset="pascal_voc_seg" \
--training_number_of_steps="200000" \
--fine_tune_batch_norm=false \
--base_learning_rate=0.000005 \
--optimizer="adam" \
--adam_learning_rate=0.0001 \
--tf_initial_checkpoint="/content/drive/MyDrive/PQR_Checkpoints_257/mnv3_small_170k_pretrained_b4_lr000005_adam_alr0001" \
--train_logdir="/content/drive/MyDrive/PQR_Checkpoints_257/mnv3_small_170k_pretrained_b4_lr000005_adam_alr0001" \
--dataset_dir="/content/tfrecord" \
--image_pooling_crop_size=257,257 \
--image_pooling_stride=4,5 \
--add_image_level_feature=1 \
--aspp_convs_filters=128 \
--aspp_with_concat_projection=0 \
--aspp_with_squeeze_and_excitation=1 \
--decoder_use_sum_merge=1 \
--decoder_filters=2 \
--decoder_output_is_logits=1 \
--image_se_uses_qsigmoid=1 \
--decoder_output_stride=8 \
--output_stride=32 \
--image_pyramid=1 \
--initialize_last_layer=true \
--last_layers_contain_logits_only=True \
```

### Segformer

The Segformer model combines the strengths of Transformers and multi-scale feature learning. In essence, Segformer is a hybrid model that capitalizes on the benefits of both self-attention mechanisms and multi-scale feature learning. It is an architecture that brings together the concept of Transformer Encoders (Bert) for the main backbone and a simple yet effective Multi-Level feature fusion technique. Segformer is designed with computational efficiency in mind, providing high-quality segmentation results even with smaller model sizes and reduced computational overhead.

Loading Official Segformer Checkpoint:

```
from transformers import TFSegformerForSemanticSegmentation

model_checkpoint = "nvidia/mit-b0"
id2label = {0: "outer", 1: "hair", 2: "face", 3: "skin", 4: "clothes", 5: "accessories"}
label2id = {label: id for id, label in id2label.items()}
num_labels = len(id2label)

model = TFSegformerForSemanticSegmentation.from_pretrained(
    model_checkpoint,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
    ignore_mismatched_sizes=True,
)
lr = 0.00006
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer)
```

Custom Checkpoint (Optional):

```
model.load_weights( "/content/drive/MyDrive/Trained-Models/1-66mb/Epoch15/Lips.ckpt")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Callbacks:

```
from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=1)
    pred_mask = tf.expand_dims(pred_mask, -1)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for sample in dataset.take(num):
            images, masks = sample["pixel_values"], sample["labels"]
            masks = tf.expand_dims(masks, -1)
            pred_masks = model.predict(images).logits
            images = tf.transpose(images, (0, 2, 3, 1))
            display([images[0], masks[0], create_mask(pred_masks)])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ]
        )


class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        clear_output(wait=True)
        show_predictions(self.dataset)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))
```

Training:

```
# Increase the number of epochs if the results are not of expected quality.
epochs = 5

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[DisplayCallback(test_ds)],
    epochs=epochs,
)
```

### Custom model

#### UNET

UNET is a convolutional neural network that was developed for biomedical image segmentation at the Computer Science Department of the University of Freiburg, Germany. The network is based on the fully convolutional network and its architecture was modified and extended to work with fewer training images and to yield more precise segmentations. UNET is built upon the architecture of a Fully Convolutional Network (FCN), but differs in the sense that it consists of a contracting path (encoder) and an expansive path (decoder).

Model Architecture:

```
base_model = tf.keras.applications.MobileNetV2(input_shape= [256, 256, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    #'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

#down_stack = quantize_model(down_stack)
down_stack.trainable = False
down_stack.summary()

up_stack = [
    pix2pix.upsample(128, 3),  # 8x8 -> 16x16
    pix2pix.upsample(64, 3),  # 16x16 -> 32x32
    pix2pix.upsample(32, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[256, 256, 3])

  # Downsampling through the model
  skips = down_stack(inputs)
  x = skips[-1]
  skips = reversed(skips[:-1])

  # Upsampling and establishing the skip connections
  for up, skip in zip(up_stack, skips):
    x = up(x)
    concat = tf.keras.layers.Concatenate()

    x = concat([x, skip])

  # This is the last layer of the model
  last = tf.keras.layers.Conv2DTranspose(
      filters=output_channels, kernel_size=3, strides=2,
      padding='same')  #64x64 -> 128x128

  x = last(x)
  return tf.keras.Model(inputs=inputs, outputs=x)
```

Compilation:

```
OUTPUT_CLASSES = 3

model = unet_model(output_channels=OUTPUT_CLASSES)

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Loading Checkpoints (Optional):

```
# Epoch 5 of removed512
#checkpoint_path = "/content/drive/MyDrive/Trained-Models/ConfigTest/Lips.ckpt"
checkpoint_path = "/content/drive/MyDrive/Trained-Models/1-66mb/Epoch30/Lips.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_path,
                                                save_weights_only=True,
                                                verbose= 1)

model.load_weights( "/content/drive/MyDrive/Trained-Models/1-66mb/Epoch15/Lips.ckpt")
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
```

Training:

```
class DisplayCallback(tf.keras.callbacks.Callback):
  def on_epoch_end(self, epoch, logs=None):
    clear_output(wait=True)
    show_predictions()
    print ('\nSample Prediction after epoch {}\n'.format(epoch+1))

def cal_steps(num_images, batch_size):
   # calculates steps for generator
   steps = num_images // batch_size

   # adds 1 to the generator steps if the steps multiplied by
   # the batch size is less than the total training samples
   return steps + 1 if (steps * batch_size) < num_images else steps

EPOCHS = 5
#VAL_SUBSPLITS = 5
#STEPS_PER_EPOCH = cal_steps( 22832, 64)
#VALIDATION_STEPS = cal_steps( 5708, 64)

STEPS_PER_EPOCH = cal_steps( 4053, 64)
VALIDATION_STEPS = cal_steps( 1013, 64)

model_history = model.fit(train_batches, epochs=EPOCHS,
                          steps_per_epoch=STEPS_PER_EPOCH,
                          validation_steps=VALIDATION_STEPS,
                          validation_data=test_batches,
                          callbacks=[DisplayCallback(), cp_callback])
```

#### Modified Vision Transformer

The Vision Transformer (ViT) is a recent advancement in the field of computer vision, it employs transformer architectures, which were originally designed for natural language processing tasks, for image recognition tasks. Our custom model modifies the traditional ViT by including additional convolutional layers and skip connections. These modifications allow the transformer to better capture the spatial hierarchies in the image, and the skip connections ensure a more effective backward propagation of gradients, enhancing the model's learning capability.

Model Architecture:

```
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow.keras import layers as L

from tensorflow.keras.utils import plot_model


def convBlock (priorNode, LAYER1= (96, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (64, (1,1), (1,1))):
  conv2d_1 = L.Conv2D(LAYER1[0], LAYER1[1], strides= LAYER1[2] , padding='same', use_bias=True, activation='relu')(priorNode)
  depth_conv2d_1 = L.DepthwiseConv2D(LAYER2[0], strides=LAYER2[1], padding='same', use_bias=True, activation='relu')(conv2d_1)
  conv2d_2 = L.Conv2D(LAYER3[0], LAYER3[1], strides=LAYER3[2], padding='same', use_bias=True)(depth_conv2d_1)
  return conv2d_2

def reshapeTransposeBlock(priorNode, RESHAPE=(8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3]):
    print("HEY")
    # print(priorNode.shape)
    print(RESHAPE)
    batch_size = tf.shape(priorNode)[0]
    reshape = tf.reshape(priorNode, RESHAPE)
    transpose = tf.transpose(reshape, perm=TRANSPOSE_PERM)
    print(transpose.shape)
    return transpose


def transposeReshapeBlock (priorNode, RESHAPE=(8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3]):
  transpose = tf.transpose(priorNode, perm=TRANSPOSE_PERM)
  batch_size = tf.shape(transpose)[0]
  reshape = tf.reshape(transpose, RESHAPE)
  return reshape

def largeBlock(priorNode, MUL=128):
  # Block 2-1
  block_2_1 = tf.multiply(priorNode, priorNode)  # Multiply with priorNode itself
  block_2_1 = tf.add(block_2_1, block_2_1)  # Add the result with itself

  # Block 2-2
  block_2_2 = L.Conv2D(1, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(block_2_1)

  batch_size = tf.shape(block_2_2)[0]
  if MUL == 128:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 1, 64))
  elif MUL == 192:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 1, 16))

  block_2_2 = L.Softmax(axis=-1)(block_2_2)

  batch_size = tf.shape(block_2_2)[0]
  if MUL == 128:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 64, 1))
  elif MUL == 192:
    block_2_2 = tf.reshape(block_2_2, (batch_size, 16, 16, 1))

  # Block 2-3
  block_2_3 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(block_2_1)
  block_2_3 = L.Multiply()([block_2_2, block_2_3])
  block_2_3 = L.Lambda(lambda x: tf.reduce_sum(x, axis=2, keepdims=True))(block_2_3)

  # Block 2-4
  block_2_4 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='valid', use_bias=True, activation='relu')(block_2_1)
  block_2_4 = L.Multiply()([block_2_3, block_2_4])
  block_2_4 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='valid', use_bias=True)(block_2_4)
  block_2_4 = L.Add()([block_2_4, priorNode])  # <-- Corrected typo: block2 -> priorNode

  # Block 2-5
  block_2_5 = L.Conv2D(MUL*2, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_1)
  block_2_5 = L.Conv2D(MUL, (1, 1), strides=(1, 1), padding='same', use_bias=True)(block_2_5)

  # Output
  return L.Add()([block_2_5, block_2_4])


# BUILD MODEL
def build_model(input_shape, classes = 6, bs = 4):
  # Input layer
  inputs = L.Input(input_shape)
  x = L.Conv2D(16, (3, 3), strides=(2, 2), padding='same', use_bias=True, activation='relu')(inputs)

  # Block 1 (CONV)
  x1 = convBlock(x, LAYER1= (96, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (64, (1,1), (1,1)))
  x1 = convBlock(x1, LAYER1= (384, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (128, (1,1), (1,1)))

  x1_1 = convBlock(x1, LAYER1= (768, (1,1), (1,1)), LAYER2=((3,3), (1,1)), LAYER3 = (128, (1,1), (1,1)))
  output1 = L.Add(name= "output1")([x1_1, x1])
  # batch_size = tf.shape(output1)[0]
  # output1 = tf.reshape(output1, (bs, 64, 64, 128))

  # Block 2
  x2 = convBlock(output1, LAYER1= (768, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (256, (1,1), (1,1)))
  x2 = L.Conv2D(128, (3,3), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x2)
  x2 = L.Conv2D(128, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x2)
  bs = tf.shape(x2)
  x2 = reshapeTransposeBlock(x2, RESHAPE= (8, 4, 8, 512), TRANSPOSE_PERM=[0, 2, 1, 3])
  output2 = reshapeTransposeBlock(x2, RESHAPE= (1, 64, 16, 128), TRANSPOSE_PERM=[0, 2, 1, 3])

  # Block 3
  x3 = largeBlock(output2)
  x3 = largeBlock(x3)

  x3 = transposeReshapeBlock(x3, RESHAPE= (8, 8, 4, 512), TRANSPOSE_PERM=[0, 2, 1, 3])
  x3 = transposeReshapeBlock(x3, RESHAPE= (1, 32, 32, 128), TRANSPOSE_PERM=[0, 2, 1, 3])
  output3 = L.Conv2D(256, (1, 1), strides=(1, 1), padding='same', use_bias=True, name="output3")(x3)

  # Block 4
  x4 = convBlock(output3, LAYER1= (1536, (1,1), (1,1)), LAYER2=((3,3), (2,2)), LAYER3 = (384, (1,1), (1,1)))
  x4 = L.Conv2D(192, (3,3), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x4)
  x4 = L.Conv2D(192, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x4)
  x4.shape

  x4 = reshapeTransposeBlock(x4, RESHAPE= (4, 4, 4, 768), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = reshapeTransposeBlock(x4, RESHAPE= (1, 16, 16, 192), TRANSPOSE_PERM=[0, 2, 1, 3])
  # x4.shape

  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)
  x4 = largeBlock(x4, MUL=192)

  x4 = transposeReshapeBlock(x4, RESHAPE= (4, 4, 4, 768), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = transposeReshapeBlock(x4, RESHAPE= (1, 16, 16, 192), TRANSPOSE_PERM=[0, 2, 1, 3])
  x4 = L.Conv2D(384, (1,1), strides= (1,1) , padding='same', use_bias=True)(x4)
  x4 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(x4)
  output4 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True, name= "output4")(x4)

  # Block 5
  x5 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(output3)
  x5 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x5)

  x5_output4 = tf.image.resize(output4, size=(32, 32), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output5 = L.Add(name= "output5")([x5, x5_output4])

  # Block 6
  x6 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(output1)
  x6 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x6)

  x6_output5 = tf.image.resize(output5, size=(64, 64), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output6 = L.Add(name= "output6")([x6, x6_output5])

  # Block 7
  x7 = L.DepthwiseConv2D((1,1), strides=(1,1), padding='same', use_bias=True)(x)
  x7 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x7)

  x7_output6 = tf.image.resize(output6, size=(128, 128), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
  output7 = L.Add(name= "output7")([x7, x7_output6])

  # Block 8 (No longer need resize nearest neighbor)
  x8 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output7)
  x8 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x8)

  x8_output6 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output6)
  x8_output6 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x8_output6)
  x8_output6 = tf.image.resize(x8_output6, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output8 = L.Add(name= "output8")([x8, x8_output6])

  # Block 9
  # x9 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output8)
  # x9 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x9)

  x9_output5 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output5)
  x9_output5 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x9_output5)
  x9_output5 = tf.image.resize(x9_output5, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output9 = L.Add(name= "output9")([output8, x9_output5])

  # Block 10
  # x10 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output9)
  # x10 = L.Conv2D(64, (1,1), strides= (1,1) , padding='same', use_bias=True)(x10)

  x10_output4 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output4)
  x10_output4 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x10_output4)
  x10_output4 = tf.image.resize(x10_output4, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
  output10 = L.Add(name= "output10")([output9, x10_output4])

  # Block 11 (Head)
  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(output10)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  x11 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True, activation='relu')(x11)
  x11 = L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)

  # block_2_2 = L.Softmax(axis=-1)(block_2_2)
  x11 = L.Conv2DTranspose(32, (2,2), strides=(2,2), padding='same', use_bias=True)(x11)

  # output11 = L.Conv2D(NUM_CLASSES, (1,1), strides= (1,1) , padding='same', use_bias=True)(x11)
  # output11 = Conv2D(num_classes, 1, padding="same")(x11)
  output11 = L.Conv2D(NUM_CLASSES, (1,1), strides= (1,1), padding='same', use_bias=True, activation='softmax')(x11)


  # Build model
  return tf.keras.Model(inputs=inputs, outputs=output11)


INPUT_SHAPE = (256, 256, 3)
NUM_CLASSES = 6
BATCH_SIZE = 1
model = build_model(INPUT_SHAPE, NUM_CLASSES, BATCH_SIZE)
```

Compilation:

```
lr = 0.00000001
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.load_weights( "/content/drive/MyDrive/Reverse_Engineer_Mediapipe/Segmentation/Test15_Normalized_LastLayer-NoActivation/class6.ckpt")
model.compile(
        loss="categorical_crossentropy",
        optimizer=tf.keras.optimizers.Adam(lr),
        metrics=['accuracy']
)
```

Checkpoints (Optional):

```
checkpoint_path = "/content/drive/MyDrive/Reverse_Engineer_Mediapipe/Segmentation/Test16_Normalized_LastLayer-NoActivation/class6.ckpt"
checkpoint_dir = os.path.dirname(checkpoint_path)

# Checkpoint callback
cp_callback = tf.keras.callbacks.ModelCheckpoint( checkpoint_path,
                                                save_weights_only=True,
                                                verbose= 1)
```

Callbacks:

```
from IPython.display import clear_output


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    mask_values = [0, 1, 2, 3, 4, 5]  # Adjust the mask values based on your dataset
    pred_mask = tf.gather(mask_values, pred_mask)
    pred_mask = tf.expand_dims(pred_mask, axis=-1)  # Expand dimensions along the channel axis
    # print(" PRED MASK SHAPE: ", pred_mask.shape)
    return pred_mask[0]


def show_predictions(dataset=None, num=1):
    if dataset:
        for sample in dataset.take(num):
            images, masks = sample[0], sample[1]
            # masks = tf.expand_dims(masks, -1)
            pred_masks = model.predict(images)
            # print(" PRED MASK SHAPE: ", pred_masks[0].shape)
            # images = tf.transpose(images, (0, 2, 3, 1))
            display([images[0], masks[0], pred_masks[0]])
    else:
        display(
            [
                sample_image,
                sample_mask,
                create_mask(model.predict(tf.expand_dims(sample_image, 0))),
            ]
        )
# for samples in train_ds.take(2):
#     sample_image, sample_mask = samples[0], samples[1]
#     display([sample_image[0], sample_mask[0]])

class DisplayCallback(tf.keras.callbacks.Callback):
    def __init__(self, dataset, **kwargs):
        super().__init__(**kwargs)
        self.dataset = dataset

    def on_epoch_end(self, epoch, logs=None):
        # clear_output(wait=True)
        show_predictions(self.dataset, 5)
        print("\nSample Prediction after epoch {}\n".format(epoch + 1))

```

Training:

```
# Increase the number of epochs if the results are not of expected quality.
epochs = 20

history = model.fit(
    train_ds,
    validation_data=test_ds,
    callbacks=[DisplayCallback(test_ds),cp_callback],
    epochs=epochs,
)
```

## Factors that influences training of Image Segmentation

### Training Data

The quality of the training data plays a crucial role. It should be diverse and representative of the types of images that the model will encounter in the real world. Also, each object that the model should recognize must be properly labeled in the training data.

#### Image Artifacts

To solve problem with image mask's artifacts on the edges, use Nearest Neighbor when resizing. Here's a sample code snippet:

```
input_mask = tf.image.resize(input_mask, (image_size, image_size), method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
```

#### Corrupted Training Data

To check for problem's in the dataset, iterate through the dataset and look for NaN. Here's a sample code snippet:

```
# Check for NaN values in train_ds
for input_image, input_mask in train_ds:
    assert not np.any(np.isnan(input_image)), "Input image data contains NaN values."
    assert not np.any(np.isnan(input_mask)), "Input mask data contains NaN values."

# Check for NaN values in test_ds
for input_image, input_mask in test_ds:
    assert not np.any(np.isnan(input_image)), "Input image data contains NaN values."
    assert not np.any(np.isnan(input_mask)), "Input mask data contains NaN values."
```

#### Original Image and Masks Mismatch

There are times when you are loading the dataset where the original image and its mask doesn't align. It could be that you load a different mask for a certain original image. To check it, here's a sample code snippet:

```
print(f" 1000th image: {image_paths[1000]} and mask: {mask_paths[1000]}")
print(f" Length image: {len(image_paths)} and mask: {len(mask_paths)}")
```

If you find a mismatch problem, modify your dataset loading.

#### Wrong labelling

When preparing your training data, check whether the pixels were allocated the right labels. It could be that your supposed image pixel which should have been class 1 was wrongly labelled as class 6 or 7 or 2 or whatever.

### Model Architecture

Different architectures will yield different results, so choosing the right one is key. For instance, a deeper model might be able to capture more complex features but may also be more prone to overfitting.

#### Number of Layers

The number of layers often determine how capable a model is when doing its task. However, you shouldn't just add and add labels. Carefully remove and edit layers to see what suits you best

#### Connections of Layers

When we are dealing with complex model architectures, the connection of the layers are very important. It will determine where a layer's input comes from and where its output go to.

Here's a sample code snippet of connecting Layers:

```
x10_output4 = L.DepthwiseConv2D((3,3), strides=(1,1), padding='same', use_bias=True)(output4)
x10_output4 = L.Conv2D(64, (1,1), strides= (1,1) , padding='valid', use_bias=True)(x10_output4)
x10_output4 = tf.image.resize(x10_output4, size=(128, 128), method=tf.image.ResizeMethod.BILINEAR)
output10 = L.Add(name= "output10")([output9, x10_output4])
```

On the given sample, we can see that we are taking the output of the layer named "output4" and connecting it to our convolutional layers which were then resized. After that, we connect the output of the resized layer to the layer named "output9" through an add layer. What this essentially does is we make a skip connection. The skip connection will connect 2 or more layers that are far from each other.

#### Layer Parameters

The layer parameters will determine how a layer will function. Here's a sample code snippet:

Sample 1:

```
L.Conv2D(32, (1,1), strides= (1,1) , padding='same', use_bias=True, activation='relu')(x11)
```

Sample 2:

```
L.Conv2D(32, (1,1), strides= (1,1) , padding='valid', use_bias=False, activation='relu')(x11)
```

The two sample might look the same if you don't pay close attention. But the layer will have vastly different outputs.

For one, the first layer uses a bias which will add a weight to the convolution. Furthermore, it also uses "SAME" as padding, what this will do is it will generate zeros if the stride overflows.

For the second sample, it doesn't use a bias which will make its output more raw. Meaning, we don't use attention to the model's neurons. Additionally, it also uses "VALID" padding, this is called no padding. As the name suggest, we don't generate zeros to deal with overflows in strides.

### Hyperparameters

The choice of hyperparameters can also greatly affect the training process. This includes the learning rate, batch size, the type and rate of regularization, and the number of training epochs.

#### Learning rate

Different learning rate will certainly affect a model training's ability to converge to the lowest point in the gradient.

A good rule that I follow is to lower the learning rate as the training progresses. This is also called learning rate decay. What this essentially does is prevent our model from getting stuck in a local minima or prevent it from overshooting.

Start from a fairly large learning rate, and if the training loss is no longer going down or it is very slow and goes up and down, then lower your learning rate.

#### Optimizer

The optimizer will also determine how we move down the gradient. As such, experiment on different optimizers. But mostly, Adam optimizer is pretty well rounded and great. So use it as default if you don't know where to start

### Loss Function

The choice of loss function can influence how well the model learns from the training data. Some commonly used loss functions in segmentation tasks are Binary Cross Entropy, Dice Loss, and Jaccard/Intersection over Union (IoU) loss.

The compilation loss parameter will determine how we will compute the loss during training. It will also determine what kind of training data we can take.

An example of this is using Categorical Cross Entropy vs Sparse Cross Entropy. It sounds similar, but the sparse cross entropy can process image masks that is (batch_size, Height, Width). On the other hand, in Categorical Cross Entropy, we will need to process the mask image to one hot encoding, this will mean that the training image mask must be (batch_size, Height, Width, number_of_classes)

### Data Augmentation

Data augmentation expands the training dataset with modified versions of the existing images. This can include rotation, scaling, flipping, and cropping. This process can help the model generalize better to new data.

However, there are also times where data augmentation becomes bad. An example of this is if we generate too much augmented greyscaled images. In that case, I noticed that the model loses its ability to use the color of the pixel todetermine its class

### Use of Pretrained Models

Using pretrained models (transfer learning) can boost performance, especially when the amount of training data is limited. These models have already learned useful features from large datasets that can be leveraged for the specific task at hand.

Always save your training every epoch or every x number of steps. This will ensure that we can use it as a checkpoint for future training. This prevents wastage of training if ever it suddenly stops or encounters problems

### Computational Resources

The amount of available memory and processing power can influence the choice of model architecture, batch size, and training time. Advanced models often require powerful GPUs and large amounts of memory.

The difference with using GPU in training is mind boggling. It is more than twice as fast than just using the CPU. As such, use GPU as much as possible if you don't want to take forever training a large model.
