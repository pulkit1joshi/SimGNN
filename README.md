# SimGNN

A Keras implementation of **SimGNN: A Neural Network Approach to Fast Graph Similarity Computation*** [Link](https://arxiv.org/abs/1808.05689).

![image](https://user-images.githubusercontent.com/42002993/95562734-2c25fc80-0a3a-11eb-9438-d0b1c7c49d63.png)

# Abstract 
Graph similarity search is among the most important graph-based applications, e.g. finding the chemical compounds that are most similar to a query compound. Graph similarity/distance computation, such as Graph Edit Distance (GED) and Maximum Common Subgraph (MCS), is the core operation of graph similarity search and many other applications, but very costly to compute in practice. Inspired by the recent success of neural network approaches to several graph applications, such as node or graph classification, we propose a novel neural network based approach to address this classic yet challenging graph problem, aiming to alleviate the computational burden while preserving a good performance. The proposed approach, called SimGNN, combines two strategies. First, we design a learnable embedding function that maps every graph into an embedding vector, which provides a global summary of a graph. A novel attention mechanism is proposed to emphasize the important nodes with respect to a specific similarity metric. Second, we design a pairwise node comparison method to sup plement the graph-level embeddings with fine-grained node-level information. Our model achieves better generalization on unseen graphs, and in the worst case runs in quadratic time with respect to the number of nodes in two graphs. Taking GED computation as an example, experimental results on three real graph datasets demonstrate the effectiveness and efficiency of our approach. Specifically, our model achieves smaller error rate and great time reduction compared against a series of baselines, including several approximation algorithms on GED computation, and many existing graph neural network based models. Our study suggests SimGNN provides a new direction for future research on graph similarity computation and graph similarity search.


Medium article on SimGNN: [Link](https://medium.com/swlh/simgnn-56420a66fa37).

*This includes only upper layer ( Attension mechanism implementation )

Tensorflow implementation from author: [Link](https://github.com/yunshengb/SimGNN) 

Another Implementation (This implementation is inspired from this repo): [Link](https://github.com/benedekrozemberczki/SimGNN)

Keras NTL: [Link](https://github.com/dapurv5/keras-neural-tensor-layer)
