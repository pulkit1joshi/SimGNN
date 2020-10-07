import tensorflow as tf
from tensorflow import keras
import argparse

# Attention layer for keras
# You just need to define the call and build in keras and backprop is taken care by the keras


class Attention(keras.layers.Layer):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

    def build(self, input_shape):
        self.weights_att = self.add_weight(
        shape=(self.args.filters_3,self.args.filters_3),
        initializer="random_normal",
        trainable=True)
        super(Attention, self).build(input_shape);

    def call(self, embedding):
        printshapes = True
        g_input = keras.backend.dot(embedding, self.weights_att)
        global_context = keras.backend.mean(g_input, axis=0)
        nl_global_context = tf.keras.activations.tanh(global_context)
        nl_global_context = tf.reshape(nl_global_context, [1,-1])
        sig_scores = tf.keras.activations.sigmoid(tf.matmul(embedding, keras.backend.transpose(tf.convert_to_tensor(nl_global_context))))
        embedd = keras.backend.transpose(tf.matmul(keras.backend.transpose(embedding), sig_scores))
        if printshapes == True:
            print("Context before mean is:                ", g_input.shape)
            print("Emedding-Shape in attention module is: ", embedding.shape)
            print("Global Context shape is:               ", nl_global_context.shape)
            print("Scores shape is:                       ", sig_scores.shape)
            print("Embedding shape is:                    ", embedd.shape)
        return nl_global_context


"""
Current Output is : 

Context before mean is:                 (11, 3)
Emedding-Shape in attention module is:  (11, 3)
Global Context shape is:                (1, 3)
Scores shape is:                        (11, 1)
Embedding shape is:                     (1, 3)
Context before mean is:                 (11, 3)
Emedding-Shape in attention module is:  (11, 3)
Global Context shape is:                (1, 3)
Scores shape is:                        (11, 1)
Embedding shape is:                     (1, 3)
Model: "sequential"
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
attention (Attention)        (1, 3)                    9         
=================================================================
Total params: 9
Trainable params: 9
Non-trainable params: 0

"""


"""
This comment is for checking and model creation for above layer will be removed layer

from keras.models import Sequential 
from keras.layers import Dense 

parser = argparse.ArgumentParser(description="Run SimGNN.")
parser.add_argument("--filters-3",
                        type=int,
                        default=3,
	                help="Filters (neurons) in 3rd convolution. Default is 32.")

model = Sequential() 
model.add(Attention(parser.parse_args())) 
y = model(tf.Variable([[-0.1673, -0.0144, -0.0100],
        [-0.1547, -0.0147, -0.0103],
        [-0.1414, -0.0155, -0.0064],
        [-0.1683, -0.0137, -0.0104],
        [-0.1627, -0.0088, -0.0142],
        [-0.1891, -0.0147, -0.0127],
        [-0.1438, -0.0139, -0.0089],
        [-0.1539, -0.0147, -0.0103],
        [-0.1445, -0.0121, -0.0104],
        [-0.1667, -0.0154, -0.0098],
        [-0.1686, -0.0125, -0.0133]]))

model.summary()
"""
"""
Uncomment to check working of the activation


x = Attention(parser.parse_args())

y = x(tf.Variable([[-0.1673, -0.0144, -0.0100],
        [-0.1547, -0.0147, -0.0103],
        [-0.1414, -0.0155, -0.0064],
        [-0.1683, -0.0137, -0.0104],
        [-0.1627, -0.0088, -0.0142],
        [-0.1891, -0.0147, -0.0127],
        [-0.1438, -0.0139, -0.0089],
        [-0.1539, -0.0147, -0.0103],
        [-0.1445, -0.0121, -0.0104],
        [-0.1667, -0.0154, -0.0098],
        [-0.1686, -0.0125, -0.0133]]))
print(y)


"""