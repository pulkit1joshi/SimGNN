# SimGNN

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-edit-distance-computation-via-graph/graph-similarity-on-imdb)](https://paperswithcode.com/sota/graph-similarity-on-imdb?p=graph-edit-distance-computation-via-graph)
[![codebeat badge](https://codebeat.co/badges/8678ae0a-67d3-423b-830d-050ed726e4eb)](https://codebeat.co/projects/github-com-pulkit1joshi-simgnn-main)
[![Maintenance](https://img.shields.io/badge/Maintained%3F-no-red.svg)](https://GitHub.com/Naereen/StrapDown.js/graphs/commit-activity)


Keras implementation of **SimGNN: A Neural Network Approach to Fast Graph Similarity Computation*** 

![image](https://user-images.githubusercontent.com/42002993/95562734-2c25fc80-0a3a-11eb-9438-d0b1c7c49d63.png)

*This includes only upper layer ( Attention mechanism implementation )

## Abstract 
Graph similarity search is among the most important graph-based applications, e.g. finding the chemical compounds that are most similar to a query compound. Graph similarity/distance computation, such as Graph Edit Distance (GED) and Maximum Common Subgraph (MCS), is the core operation of graph similarity search and many other applications, but very costly to compute in practice. Inspired by the recent success of neural network approaches to several graph applications, such as node or graph classification, we propose a novel neural network based approach to address this classic yet challenging graph problem, aiming to alleviate the computational burden while preserving a good performance. The proposed approach, called SimGNN, combines two strategies. First, we design a learnable embedding function that maps every graph into an embedding vector, which provides a global summary of a graph. A novel attention mechanism is proposed to emphasize the important nodes with respect to a specific similarity metric. Second, we design a pairwise node comparison method to sup plement the graph-level embeddings with fine-grained node-level information. Our model achieves better generalization on unseen graphs, and in the worst case runs in quadratic time with respect to the number of nodes in two graphs. Taking GED computation as an example, experimental results on three real graph datasets demonstrate the effectiveness and efficiency of our approach. Specifically, our model achieves smaller error rate and great time reduction compared against a series of baselines, including several approximation algorithms on GED computation, and many existing graph neural network based models. Our study suggests SimGNN provides a new direction for future research on graph similarity computation and graph similarity search.

## Resources

Paper can be found here: [Link](https://arxiv.org/abs/1808.05689). </br>
Tensorflow implementation from author: [Link](https://github.com/yunshengb/SimGNN)  </br>
Another Implementation: [Link](https://github.com/benedekrozemberczki/SimGNN) </br>
Medium article : [Link](https://medium.com/swlh/simgnn-56420a66fa37?source=post_stats_page-------------------------------------) </br>

## Results

![image](https://github.com/pulkit1joshi/SimGNN/blob/main/Training.png)

Above results on test data, that is error of order 8 x 10<sup>-3</sup>, were obtained when trained on synthetic data (11k graph pairs that is 0.224 times original data) (200 epochs) with training loss of order 10<sup>-4</sup>.

|    Method    |     Test Error (Synth Data)      |   Test Error (SimGNN paper)  |
|  ---------   | ------------------- |----------------|
|  SimpleMean  |       1.15x10<sup>-2</sup>         |   3.749x10<sup>-3</sup>  |
| SimGNN (Att) |       8.80x10<sup>-3</sup>         |   1.455x10<sup>-3</sup>  |
|  Difference  |       0.0027        |   0.002294     |
|    Ratio     |       1.307         |     2.5766        |
