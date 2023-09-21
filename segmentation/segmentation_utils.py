import tensorflow as tf
import numpy as np
import cv2

def classify(x):
    return np.argmax(x,axis=-1).astype(np.uint8)

def preprocess(x, input_shape):
    x = cv2.resize(x, (input_shape[2], input_shape[1]), interpolation=cv2.INTER_NEAREST)
    x = cv2.cvtColor(x, cv2.COLOR_BGR2RGB)
    x = x / 255.0
    x = np.array(x, dtype="float32")
    x = np.expand_dims(x, axis=0)
    return x
