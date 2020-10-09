""" Main SimGNN model """

from tensorflow import keras
from tensorflow.keras import layers
from keras_gcn import GraphConv
from keras.models import Model
from keras.layers import Input
import numpy as np
from tqdm import tqdm, trange
from custom_layers import Attention, NeuralTensorLayer
from parser import parameter_parser
from utilities import data2, convert_to_keras, process, find_loss



parser = parameter_parser()
""" 
Main model : Node-to-Node interaction not implemented.
Functional API :
Shared layers are shared_gcn1, shared_gcn2, shard_gcn3, shared_attention
"""

def main():
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

    model = Model(inputs=[inputA, GinputA, inputB, GinputB], outputs=z)
    opt = keras.optimizers.Adadelta(learning_rate=parser.learning_rate, rho=parser.weight_decay)
    model.compile(
                optimizer=opt,
                loss='mean_squared_error',
                metrics=['mean_squared_error'],
            )
    model.summary()

    """ 
    Data loading
    """

    x = data2()
    batches = x.create_batches()
    global_labels = x.getlabels()

    """
    Training the Network
    Take every graph pair and train it as a batch.
    """

    for epoch in range(0,parser.epochs):
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for graph_pair in batch:
                data = process(graph_pair)
                data = convert_to_keras(data, global_labels)
                x = np.array([ data["features_1"] ])
                y = np.array([ data["features_2"] ])
                a = np.array([ data["edge_index_1"] ])
                b = np.array([ data["edge_index_2"] ])
                model.train_on_batch([x, a, y, b], data["target"])

    """ 
    Testing over the test data
    """

    x = data2()
    test = x.gettest()
    scores = []
    g_truth = []
    for graph_pair in tqdm(test):
        data = process(graph_pair)
        data = convert_to_keras(data, global_labels)
        x = np.array([ data["features_1"] ])
        y = np.array([ data["features_2"] ])
        a = np.array([ data["edge_index_1"] ])
        b = np.array([ data["edge_index_2"] ])
        g_truth.append(data["target"])
        y=model.predict([x, a, y, b])
        scores.append(find_loss(y, data["target"]))

    norm_ged_mean = np.mean(g_truth)
    base_error = np.mean([(n-norm_ged_mean)**2 for n in g_truth])
    model_error = np.mean(scores)
    print("\nBaseline error: " +str(round(base_error, 5))+".")
    print("\nModel test error: " +str(round(model_error, 5))+".")


if __name__ == "__main__":
    main()

