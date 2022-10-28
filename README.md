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
that communicate with the server over gRPC.

#### Client

A Client is a device or process that participates in a federated training round,
by fitting the global model to its local data and then sending the updated
parameters back to the server.

In this demo, the clients are just local processes running a Python script; in a
real-world application, the clients could be a mobile application that runs on
users' smartphones. Flower provides an Android SDK and offers an
[example app](https://github.com/adap/flower/tree/main/examples/android) to help
you get started if you want to run it in a mobile setting.
#### Server


#### Strategy
