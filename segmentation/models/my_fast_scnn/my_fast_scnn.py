import tensorflow as tf

## Depthwise separable convolutional nn
# https://machinelearningmastery.com/using-depthwise-separable-convolutions-in-tensorflow/
# https://github.com/kshitizrimal/Fast-SCNN 

# parameters:
# t : expansion factor of the bottleneck block
# c : number of output channels
# n : number of times block is repeated
# s : stride parameter

def _Conv2D(inputs, kernel_size, c, n, s, relu=True):
    x = inputs
    for i in range(n):
        x = tf.keras.layers.Conv2D(filters=c, kernel_size=kernel_size, strides=s, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        if relu:
            x = tf.keras.layers.ReLU()(x)
    return x

def _DSConv(inputs, kernel_size, c, n, s):
    x = inputs
    for i in range(n):
        x = tf.keras.layers.SeparableConv2D(filters=c, kernel_size=kernel_size, strides=s, padding='same')(x)
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.ReLU()(x)
    return x

def _DWConv(inputs, kernel_size, s):
    x = inputs
    x = tf.keras.layers.DepthwiseConv2D(kernel_size=kernel_size, strides=s, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = x = tf.keras.layers.ReLU()(x)
    return x

def _bottleneck(inputs, t, c, n, s):
    
    tc = tf.keras.backend.int_shape(inputs)[-1] * t

    x = inputs
    x = _Conv2D(inputs=x, kernel_size=1, c=tc, n=1, s=1)
    x = _DWConv(inputs=x, kernel_size=3, s=s) # the stride is applied only to the first sequence of the repeating block
    x = _Conv2D(inputs=x, kernel_size=1, c=c, n=1, s=1, relu=False)

    for i in range(n-1):
        x = _Conv2D(inputs=x, kernel_size=1, c=tc, n=1, s=1)
        x = _DWConv(inputs=x, kernel_size=3, s=1)
        x = _Conv2D(inputs=x, kernel_size=1, c=c, n=1, s=1, relu=False)

    return x

# Ref : https://arxiv.org/pdf/1612.01105.pdf
def _PPM(inputs, c):

    concat_list = [inputs]
    height = tf.keras.backend.int_shape(inputs)[1]
    width = tf.keras.backend.int_shape(inputs)[2]
    
    # From the original paper:
    # Noted that the number of pyramid levels and size of each
    # level can be modified. They are related to the size of feature
    # map that is fed into the pyramid pooling layer
    # Note that the original PPM use bin_sizes = [1, 2, 3, 6]
    bin_sizes = [1, 2, 3, 6]

    # From the original paper:
    # To maintain the weight of global feature, we use 1×1
    # convolution layer after each pyramid level to reduce the dimension of context 
    # representation to 1/N of the original one if the level size of pyramid is N.
    context_filters = tf.keras.backend.int_shape(inputs)[-1]//len(bin_sizes)
    
    for bin_size in bin_sizes:
        x = tf.keras.layers.AveragePooling2D(pool_size=(height//bin_size, width//bin_size), strides=(height//bin_size, width//bin_size))(inputs)
        x = tf.keras.layers.Conv2D(filters=context_filters, kernel_size=1, padding='same')(x)
        x = tf.keras.layers.experimental.preprocessing.Resizing(height, width, interpolation='bilinear')(x)
        concat_list.append(x)

    x = tf.keras.layers.concatenate(concat_list)

    # Note: the feature map has size 256 and not 128 as it have to be. 
    # So if needed, add the following pointwise conv2D layer that is not present in the paper to have 128
    # It is actually not present neither in the paper of FAST SCNN nor in PPM
    # x = tf.keras.layers.Conv2D(filters=c, kernel_size=1, strides=1, padding='same')(x)

    return x

def _FFM(inputs_high_res, inputs_low_res, c):

    # ratio between high res and low res
    height_X = tf.keras.backend.int_shape(inputs_high_res)[1]//tf.keras.backend.int_shape(inputs_low_res)[1]
    width_X = tf.keras.backend.int_shape(inputs_high_res)[2]//tf.keras.backend.int_shape(inputs_low_res)[2]

    # high res branch
    x_high_res = inputs_high_res
    x_high_res = _Conv2D(inputs=x_high_res, kernel_size=1, c=c, n=1, s=1, relu=False)

    # low res branch
    x_low_res = inputs_low_res
    x_low_res = tf.keras.layers.UpSampling2D(size=(height_X, width_X))(x_low_res)
    x_low_res = _DWConv(inputs=x_low_res, kernel_size=3, s=1)
    x_low_res = _Conv2D(inputs=x_low_res, kernel_size=1, c=c, n=1, s=1, relu=False)

    # add high res branch and low res branch
    x = tf.keras.layers.add([x_high_res, x_low_res])
        
    return x

def _LearningToDownsample(inputs):
    x = inputs
    x = _Conv2D(inputs=x, kernel_size=3, c=32, n=1, s=2)
    x = _DSConv(inputs=x, kernel_size=3, c=48, n=1, s=2)
    x = _DSConv(inputs=x, kernel_size=3, c=64, n=1, s=2)

    return x

def _GlobalFeatureExtractor(inputs):
    x = inputs
    x = _bottleneck(inputs=x, c=64, t=6, n=3, s=2)
    x = _bottleneck(inputs=x, c=96, t=6, n=3, s=2)
    x = _bottleneck(inputs=x, c=128, t=6, n=3, s=1)
    x = _PPM(inputs=x, c=128)

    return x

def _FeatureFusionModule(inputs_high_res, inputs_low_res):
    x = _FFM(inputs_high_res=inputs_high_res, inputs_low_res=inputs_low_res, c=128)

    return x

def _Classifier(inputs, input_shape, output_channels):

    height_X = input_shape[0]//tf.keras.backend.int_shape(inputs)[1]
    width_X = input_shape[1]//tf.keras.backend.int_shape(inputs)[2]

    x = inputs
    x = _DSConv(inputs=x, kernel_size=3, c=128, n=2, s=1) # The original paper does not specify the size of the kernel
    x = _Conv2D(inputs=x, kernel_size=1, c=output_channels, n=1, s=1, relu=True) # The original paper does not specify if here ReLU is used or not
    x = tf.keras.layers.UpSampling2D(size=(height_X, width_X))(x)

    return x

def fast_scnn(output_channels:int):
    #input_shape = (1024, 2048, 3) # original input shape of the paper
    input_shape = (480, 640, 3)
    input_layer = tf.keras.layers.Input(shape=input_shape, name = 'input_layer')
    ltd_layer = _LearningToDownsample(input_layer)
    gfe_layer = _GlobalFeatureExtractor(ltd_layer)
    ffm_layer = _FeatureFusionModule(ltd_layer, gfe_layer)
    cls_layer = _Classifier(ffm_layer, input_shape, output_channels)
    dropout_layer = tf.keras.layers.Dropout(0.1)(cls_layer) # dropout prob not specified in the original paper
    output_layer = tf.keras.layers.Softmax(axis=-1)(dropout_layer)
    
    # define the model
    fast_scnn = tf.keras.Model(inputs = input_layer , outputs = output_layer, name = 'Fast_SCNN')
    
    return fast_scnn

if __name__ == "__main__":
    ## TEST ##
    model = fast_scnn(output_channels=2)
    optimizer = tf.keras.optimizers.SGD(momentum=0.9, learning_rate=0.045)
    model.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), optimizer=optimizer, metrics=['accuracy'])
    tf.keras.utils.plot_model(model, show_shapes=True)