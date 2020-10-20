""" Main SimGNN model """
from tensorflow import keras
from keras import backend as K
from keras_gcn import GraphConv
import numpy as np
from tqdm import tqdm, trange
from parser import parameter_parser
from utilities import data2, convert_to_keras, process, find_loss
from simgnn import simgnn
from custom_layers import Attention, NeuralTensorLayer
from keras.backend import manual_variable_initialization 
manual_variable_initialization(True)

parser = parameter_parser()

def train(model, x):
    batches = x.create_batches()
    global_labels = x.getlabels()
    """
    Training the Network
    Take every graph pair and train it as a batch.
    """
    t_x = x
    last=0
    for epoch in range(0,parser.epochs):
        p=0
        for index, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            for graph_pair in batch:
                data = process(graph_pair)
                data = convert_to_keras(data, global_labels)
                x = np.array([ data["features_1"] ])
                y = np.array([ data["features_2"] ])
                a = np.array([ data["edge_index_1"] ])
                b = np.array([ data["edge_index_2"] ])
                p = model.train_on_batch([x, a, y, b], data["target"])
        if epoch%(parser.saveafter) == 0:
                print("Train Error:")
                print(p)
                last=z
                model.save("train")
                model.save_weights("xweights")
                
            #print("saved")

def traintest(model, x, batch):
    global_labels = x.getlabels()
    test =batch
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
    model_error = np.mean(scores)
    print("\nModel test error: " +str(round(model_error, 5))+".")
    return model_error

def test(model, x):
    global_labels = x.getlabels()
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
    model_error = np.mean(scores)
    print("\nModel test error: " +str(round(model_error, 5))+".")
    return model_error

def main():
    model = simgnn(parser);
    opt = keras.optimizers.Adadelta(learning_rate=parser.learning_rate, rho=parser.weight_decay)
    #opt = keras.optimizers.Adam(learning_rate=parser.learning_rate)
    model.compile(
                optimizer=opt,
                loss='mse',
                metrics=[keras.metrics.MeanSquaredError()],
            )
    model.summary()
    model.save("train")
    """"
    x : Data loading
    train used to train
    test over the test data
    """
    model = keras.models.load_model('train', custom_objects={'Attention': Attention, 'NeuralTensorLayer': NeuralTensorLayer, "GraphConv": GraphConv})
    K.set_value(model.optimizer.lr, parser.learning_rate)
    K.set_value(model.optimizer.decay, parser.weight_decay)
    x = data2()
    z = test(model, x)
    train(model, x)
    test(model, x)


if __name__ == "__main__":
    main()

