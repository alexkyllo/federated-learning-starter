# federated-learning-starter

A demo and starter project for federated learning with the Flower framework

## Prerequisites

- Have Python installed (optionally via [Anaconda](https://www.anaconda.com/products/distribution))
- Know how to install Python packages using `pip install` or `conda install`

## Setup

`pip install -r requirements.txt`

## Flower

Flower [[website](https://flower.dev/)]
[[github repo](https://github.com/adap/flower)] is a lightweight Python
framework for federated learning. It's framework-agnostic, so it works with
TensorFlow, PyTorch, XGBoost, scikit-learn, etc.

### Flower Architecture

Flower at the highest level consists of a central server and one or more clients
that communicate with the server to send parameters (weights) over gRPC.

There are three main modules / classes that you should know well in order to use
Flower: `Client`, `Server`, and `Strategy`.

#### Client

A _client_ is a device or process that participates in a federated training
round, by fitting the global model to its local data and then sending the
updated parameters back to the server.

In this demo, the clients are just local processes running a Python script; in a
real-world application, the clients could be a mobile application that runs on
users' smartphones. Flower provides an Android SDK and offers an
[example app](https://github.com/adap/flower/tree/main/examples/android) to help
you get started if you want to run it in a mobile setting.

#### Server

A _server_ in Flower is a central device or process that waits until a specified
minimum number of clients have connected, initializes a model, sends the
parameters to a sample of clients, waits to receive their updated weights, and
then averages them together using a _strategy_ algorithm. This constitutes one
_round_ of training.

#### Strategy

A _strategy_ in Flower is an algorithm used to combine the results of parameter
updates from multiple clients, to obtain the new parameters for the fitted
model. There are many strategies, the simplest is `FedAvg`, which is an average
of each parameter weighted by the number of training examples.

It has been demonstrated in the literature that `FedAvg` can perform poorly when
the training data is not independently and identically distributed across
clients, and algorithms to address this _non-IID_ data issue are an active
research area.
