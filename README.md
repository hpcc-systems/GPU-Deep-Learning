# GPU Accelerated Neural Networks on HPCC Systems Platform
Bundle for building, training, and consuming neural networks on HPCC Systems with GPU acceleration. 

## Description
Large nueral networks typically train on very large datasets. 
The size of the data combined with the size and complexity of neural networks result in large computational requirements. 
Until now, HPCC Systems is primarily a CPU based system which can result in neural network training times that are impractcally long.
With the use of modern GPUs, the training time can be drastically reduced over using CPUs alone.

## Getting Started

### Requirements
You must have a compatible NVIDIA GPU for the GPU acceleration to work, however this bundle will work on CPU alone, albeit significantly slower. 
CPUs are significanlty slower when it comes to training neural networks. 
If you do not have GPU/s available, see [Distributed-Deep-Learning](https://github.com/hpcc-systems/Distributed-Deep-Learning) for use on CPU only instances.

There is a AWS AMI that was created for use with this bundle. 
It is generated using [this](https://github.com/xwang2713/cloud-image-build). 
This produces an image with HPCC Systems Platform Community edition, version 7.2.14 pre-installed as well as all other requirements for this bundle, including CUDA version 10.0. 
The image is designed to run on Amazon's [P2](https://aws.amazon.com/ec2/instance-types/p2/) or [P3](https://aws.amazon.com/ec2/instance-types/p3/) machines.

### Included Training Data
Included are some popular [Datasets](Datasets/data_files) used in experimenting with neural networks slightly modified for easy spraying on the HPCC Systems Platform. 
The scripts used to generate the modified datasets is also included, which uses the original datasets as a staring point.

* [MNIST](Datasets/data_files/mnist), see [reference](http://yann.lecun.com/exdb/mnist/)
* [Fashion MNIST](data_files/fashion_mnist), see [reference](https://github.com/zalandoresearch/fashion-mnist)
* [IMDB](Datasets/data_files/imdb)
* [Boston Housing](Datasets/data_files/boston_housing), see [reference](https://doi.org/10.1016/0095-0696(78)90006-2)
* [Reuters](Datasets/data_files/reuters)

#### Spraying
Spray in the following way, with similar names, for the examples to work without modification.

* MNIST: Fixed size = 785
	* Train: mnist::train
	* Test: mnist::test
* Fashion MNIST: Fixed size = 785
	* Train: fashionmnist::train
	* Test: fashionmnist::test
* IMDB: CSV
	* Train: imdb::train
	* Test: imdb::test
* Boston Housing: CSV
	* Train: bostonhousing::train
	* Test: bostonhousing::test
* Reuters: CSV
	* Train: reuters::train
	* Test: reuters::test

### Examples
Included in this bundle are some [examples](examples/), found in the examples directory. It is recomended to start with an [MLP trained on MNIST](examples/mnist_mlp.ecl)

## Defining a Neural Network
There are three different ways you can build neural networks with this bundle.

All three methods provide an example on how to build a 10-class MLP and train on the MNIST dataset.
The MLP has 2 hidden layers of size 512, each using the "relu" activation fucntion, and a 10 class output layer with "softmax" as the activation.
All train for 20 epochs, using batch size of 128, and it is expected to achieve a Test accuracy of roughly 98%. The CNN is slightly different and gets 99% accuracy.

### Using only ECL
The first and easiest way is to use one of the predfined architecture types and define how many layers and neurons you want. You can also hyperparameter tune using this method.
For example, you can choose to build an MLP, and then define the architecture specifics, by passing in a SET of INTEGER, such as number of hidden layers, number of neurons in the each layer,
the activation functions. This is done all in ECL. 
* See the [MLP](examples/mnist_mlp.ecl) ECL example.

The second method gives you more control over the arhictecture. Use the "model.add" part of the module to iteratively add layers until the desired architecture is realized.

* See the MLP [example](examples/mlp_add_layers.ecl).
* See the CNN [example](examples/cnn_add_layers.ecl).

### Custom Architectures in Python
The thrid method is the most complex in that it requires you to have working knowledge of Python and the underlying library.
You can define a "custom" neural network architecture using Keras or PyTorch and train via HPCC/ECL.
Using this approach, any keras defined model will easily be traininable and consumable on your HPCC Systems cluster.

* See the [MLP](examples/custom_tensorflow_mlp.ecl) example.
* See the [CNN](examples/custom_tensorflow_cnn.ecl) example.


## Training Data

## Training and Testing an NN model

## Using the Model (Inference)
If you persist the model and use the model.predict() fucntion, you can use a trained model to make predictions on any incoming data on your HPCC cluster, as long as it has been prepared to the same format as the training data.
The prediction method outputs the result/s in one-hot-encoded form in the following format:

```
oneHot := RECORD
  SET of INTEGER class;
END;
```

A one-hot-encoded format for a 10 class output would be a set of 10 integers, all of which are 0, except for one. The index of the 1 will be the class that row of data was predicted to be.
i.e. a row that is predicted to belong to class 1 would be [1 0 0 0 0 0 0 0 0 0 0 ].

## TODO:
This is the planned future work that will expand upon this bundle:
* Continue adding supported Keras layers into [layers.ecl](layers.ecl)
	* Full list starts [here](https://keras.io/layers/core/)
* Make custom data handlers for importing different-formatted training data


## Author
Robert K.L. Kennedy | [GitHub](https://github.com/robertken) | [LinkedIn](https://www.linkedin.com/in/robertken/)



