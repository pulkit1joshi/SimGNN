""" Main SimGNN model """
from tensorflow import keras
import numpy as np
from tqdm import tqdm, trange
from parser import parameter_parser
from utilities import data2, convert_to_keras, process, find_loss
from simgnn import simgnn


parser = parameter_parser()

def train(model, x):
    batches = x.create_batches()
    global_labels = x.getlabels()
    """
    Training the Network
    Take every graph pair and train it as a batch.
    """
    t_x = x
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
        if epoch%(parser.saveafter) == 0:
            model.save("train")
            print("saved")

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
    """"
    x : Data loading
    train used to train
    test over the test data
    """
    model = keras.models.load_model('train')
    x = data2()
    test(model, x)
    train(model, x)
    test(model, x)
    #model.save('./models')


if __name__ == "__main__":
    main()

