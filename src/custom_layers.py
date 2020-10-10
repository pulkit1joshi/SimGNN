import tensorflow as tf
from tensorflow import keras
import scipy.stats as stats
from keras import backend as K
from keras.engine.topology import Layer

""" 
Attention mechanism as given in SimGNN
Initilizer : ( GlorotNormal ) https://keras.io/api/layers/initializers/
"""

class Attention(keras.layers.Layer):
    def __init__(self, args):
        super(Attention, self).__init__()
        self.args = args

    def build(self, input_shape):
        xavier_uniform = keras.initializers.GlorotNormal(seed=None)
        self.weights_att = self.add_weight(
        shape=(self.args.filters_3,self.args.filters_3),
        initializer=xavier_uniform,
        trainable=True)
        super(Attention, self).build(input_shape);

    def call(self, embedding):
        printshapes = False
        g_input = keras.backend.dot(embedding, self.weights_att)
        global_context = keras.backend.mean(g_input, axis=0)
        nl_global_context = keras.activations.tanh(global_context)
        nl_global_context = tf.reshape(nl_global_context, [1,-1])
        sig_scores = keras.activations.sigmoid(tf.matmul(embedding, keras.backend.transpose(tf.convert_to_tensor(nl_global_context))))
        embedd = keras.backend.transpose(tf.matmul(keras.backend.transpose(embedding), sig_scores))
        if printshapes == True:
            print("Context before mean is:                ", g_input.shape)
            print("Emedding-Shape in attention module is: ", embedding.shape)
            print("Global Context shape is:               ", nl_global_context.shape)
            print("Scores shape is:                       ", sig_scores.shape)
            print("Embedding shape is:                    ", embedd.shape)
        return embedd

"""
Github : https://github.com/dapurv5/keras-neural-tensor-layer
Credit to repo owner
"""

class NeuralTensorLayer(Layer):
  def __init__(self, output_dim, input_dim=None, **kwargs):
    self.output_dim = output_dim #k
    self.input_dim = input_dim   #d
    if self.input_dim:
      kwargs['input_shape'] = (self.input_dim,)
    super(NeuralTensorLayer, self).__init__(**kwargs)


  def build(self, input_shape):
    mean = 0.0
    std = 1.0
    # W : k*d*d
    k = self.output_dim
    d = self.input_dim
    initial_W_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(k,d,d))
    initial_V_values = stats.truncnorm.rvs(-2 * std, 2 * std, loc=mean, scale=std, size=(2*d,k))
    self.W = K.variable(initial_W_values)
    self.V = K.variable(initial_V_values)
    self.b = K.zeros((self.input_dim,))
    self.trainable_weights2 = [self.W, self.V, self.b]


  def call(self, inputs, mask=None):
    if type(inputs) is not list or len(inputs) <= 1:
      raise Exception('BilinearTensorLayer must be called on a list of tensors '
                      '(at least 2). Got: ' + str(inputs))
    e1 = inputs[0]
    e2 = inputs[1]
    batch_size = K.shape(e1)[0]
    k = self.output_dim
    # print([e1,e2])
    feed_forward_product = K.dot(K.concatenate([e1,e2]), self.V)
    # print(feed_forward_product)
    bilinear_tensor_products = [ K.sum((e2 * K.dot(e1, self.W[0])) + self.b, axis=1) ]
    # print(bilinear_tensor_products)
    for i in range(k)[1:]:
      btp = K.sum((e2 * K.dot(e1, self.W[i])) + self.b, axis=1)
      bilinear_tensor_products.append(btp)
    result = K.tanh(K.reshape(K.concatenate(bilinear_tensor_products, axis=0), (batch_size, k)) + feed_forward_product)
    # print(result)
    return result


  def compute_output_shape(self, input_shape):
    # print (input_shape)
    batch_size = input_shape[0][0]
    return (batch_size, self.output_dim)
