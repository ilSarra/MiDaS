# Reference : https://www.tensorflow.org/tutorials/images/segmentation

import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

# Define the model

# The model being used here is a modified U-Net. 
# A U-Net consists of an encoder (downsampler) and decoder (upsampler). 
# To learn robust features and reduce the number of trainable parameters, 
# use a pretrained model—MobileNetV2—as the encoder. 
# For the decoder, you will use the upsample block, 
# which is already implemented in the pix2pix example in the TensorFlow Examples repo. 
# (Check out the pix2pix: Image-to-image translation with a conditional GAN tutorial in a notebook.)

#As mentioned, the encoder is a pretrained MobileNetV2 model. 
# You will use the model from tf.keras.applications. 
# The encoder consists of specific outputs from intermediate layers in the model. 
# Note that the encoder will not be trained during the training process.

base_model = tf.keras.applications.MobileNetV2(input_shape=[480, 640, 3], include_top=False)

# Use the activations of these layers
layer_names = [
    'block_1_expand_relu',   # 64x64
    'block_3_expand_relu',   # 32x32
    'block_6_expand_relu',   # 16x16
    'block_13_expand_relu',  # 8x8
    'block_16_project',      # 4x4
]
base_model_outputs = [base_model.get_layer(name).output for name in layer_names]

# Create the feature extraction model
down_stack = tf.keras.Model(inputs=base_model.input, outputs=base_model_outputs)

down_stack.trainable = False

# The decoder/upsampler is simply a series of upsample blocks implemented in TensorFlow examples:

up_stack = [
    pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    pix2pix.upsample(64, 3),   # 32x32 -> 64x64
]

def unet_model(output_channels:int):
  inputs = tf.keras.layers.Input(shape=[480, 640, 3])

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

# Note that the number of filters on the last layer is set to the number of output_channels. 
# This will be one output channel per class.


## Usage ##

# OUTPUT_CLASSES = 3

# model = unet_model(output_channels=OUTPUT_CLASSES)
# model.compile(optimizer='adam',
#               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#               metrics=['accuracy'])