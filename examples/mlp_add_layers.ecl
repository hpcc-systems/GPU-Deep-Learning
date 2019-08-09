#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This examples demonstrates how to build an MLP model using the model.add functionality.

To define the model, iteratively call the appropriate layer function,
to define your mode. No changes to the training or infering process.

The expected accuracy on unseen test data is ~98.3%.

The training dataset is the MNIST handrwitten digit dataset, an immage classification
task.

The MLP will have an input layer of 784 (one for each pixel in the 28x28 images,
two hidden layers with 512 neurons each, and a 10 neuron output, one for each
class (digits 0 - 9). It will use the relu activation for intermediate layers
and softmax for the final output layer. The optimizer will be the RMSprop.
*/


/*Next we define the optimizer. Supported optimizers are any Keras provides. 
These are listed in this code's readme or on Keras.io.*/
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


/*
This builds an MLP with 2 hidden layers, each with 512 neurons
and the relu activation function. Its input layer is 784, and
its output layer is for 10 class classification using the 
softmax activation function.
*/
nn := model.init();
nnInputLayer := model.add.dense(nn, 'relu', 512, [784]);
nn2 := model.add.dropout(nnInputLayer,0.2);
nn3 := model.add.dense(nn2, 'relu', 512);
nn4 := model.add.dropout(nn3, 0.2);
nnOutputLayer := model.add.dense(nn4, 'softmax', 10);

OUTPUT(nnOutputLayer);

mnistTrain := Datasets.mnist.LoadTrainData();
mnistTest := Datasets.mnist.LoadTestData();


/*
Now we can start the actual training. The first line will train the model
and the second line will take the trained model and test its ability
to generalize by testing on unseen test data.

Both of these output performance metrics, first the training and validation,
second is the test performance metrics.
*/
trainedModel := model.train(nnOutputLayer, optimizer, trainingParameters, mnistTrain);
testPerformanceMetrics := model.test(trainedModel, mnistTest);
OUTPUT(trainedModel.performanceMetrics);
OUTPUT(testPerformanceMetrics);



//EXPORT add_layers := 'todo';