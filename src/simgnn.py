from tensorflow import keras
from tensorflow.keras import layers
from keras_gcn import GraphConv
from keras.models import Model
from keras.layers import Input
from custom_layers import Attention, NeuralTensorLayer
""" 
Main model : Node-to-Node interaction not implemented.
Functional API :
Shared layers are shared_gcn1, shared_gcn2, shard_gcn3, shared_attention
"""
def simgnn(parser):
    inputA = Input(shape=(None,16))
    GinputA = Input(shape=(None,None))
    inputB = Input(shape=(None,16))
    GinputB = Input(shape=(None,None))
    
    shared_gcn1 =  GraphConv(units=parser.filters_1,step_num=50, activation="relu")
    shared_gcn2 =  GraphConv(units=parser.filters_2,step_num=50, activation="relu")
    shared_gcn3 =  GraphConv(units=parser.filters_3,step_num=50, activation="relu")
    shared_attention =  Attention(parser)

    x = shared_gcn1([inputA, GinputA])
    x = shared_gcn2([x, GinputA])
    x = shared_gcn3([x, GinputA])
    x = shared_attention(x[0])

    y = shared_gcn1([inputB, GinputB])
    y = shared_gcn2([y, GinputB])
    y = shared_gcn3([y, GinputB])
    y = shared_attention(y[0])

    z = NeuralTensorLayer(output_dim=32, input_dim=32)([x, y])
    z = keras.layers.Dense(32, activation='relu')(z)
    z = keras.layers.Dense(1)(z)
    z = keras.activations.sigmoid(z)

    return Model(inputs=[inputA, GinputA, inputB, GinputB], outputs=z)