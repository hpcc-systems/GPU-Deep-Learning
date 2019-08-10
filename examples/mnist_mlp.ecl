#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This examples demonstrates how to create a Mutli Layer Perceptron (MLP), train the
model on a training dataset, test the model on unseen test data, and use the trained
model to predict on a dataset. We will train for 20 epochs with 128 size batch.

The expected accuracy on unseen test data is ~98.3%.

The training dataset is the MNIST handrwitten digit dataset, an immage classification
task.

The MLP will have an input layer of 784 (one for each pixel in the 28x28 images,
two hidden layers with 512 neurons each, and a 10 neuron output, one for each
class (digits 0 - 9). It will use the relu activation for intermediate layers
and softmax for the final output layer. The optimizer will be the RMSprop.
*/

//First let's define the hidden layers, activations and input and output shape.
hiddenLayers := [512,512];
activation := ['relu','softmax'];
IOShape := [784, 10];

//Next we define the optimizer. Supported optimizers are any Keras provides. 
//These are listed in this code's readme or on Keras.io.
optimizer := DATASET(
	[{'model', 'rmsprop'},
	{'lr', '0.001'},
	{'momentum', '0.0'},
	{'decay', '0.0'},
	{'nesterov', 'false'},
	{'rho', '0.9'},
	{'epsilon', 'none'},
	{'beta_1', '0.9'},
	{'beta_2', '0.999'},
	{'amsgrad','false'},
	{'schedule_decay', '0.004'},
	{'loss_function','categorical_crossentropy'}]
,dt.optimizerRec);


//Next we set the umber of epochs, and the batch size
trainingParameters := DATASET(
	[{'batch_size',128},
	{'epochs',20}]
, dt.parameterRec);


//We now create the actual model with the defined parameters
NN := model.mlp(hiddenLayers, activation, IOShape);


//Now that the model and parameters are defined, we can load the
//training and testing datasets. The below functions will load the
//data from a logical file and produce a numpy array, used in the
//actual computations.
mnistTrain := Datasets.mnist.LoadTrainData();
mnistTest := Datasets.mnist.LoadTestData();


/*
Now we can start the actual training. The first line will train the model
and the second line will take the trained model and test its ability
to generalize by testing on unseen test data.

Both of these output performance metrics, first the training and validation,
second is the test performance metrics.
*/
trainedModel := model.train(NN, optimizer, trainingParameters, mnistTrain);
testPerformanceMetrics := model.test(trainedModel, mnistTest);
OUTPUT(trainedModel.performanceMetrics);
OUTPUT(testPerformanceMetrics);

/*
Now that we have a neural network trained and has acceptable performance,
we can now use this model to make predictions on *new* data directly from
a logical file.

The output of the model is its prediction in one hot encoded form. Represented
as a set of integers where the length of the set is the number of classes, and
the index of the only "1" in the set is the class the model predicted.

Note: *new* this is using test data in a different format, but it demonstrates 
that if used in production, the incoming images only need to be in this format.
*/
testDS0 := CHOOSEN(DATASET('~mnist::test', dt.mnist_data_type, THOR), 10);
testDS := global(testDS0,FEW);
predictResults := model.predict(trainedModel, testDS);
OUTPUT(predictResults);
