#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This examples demonstrates how build a CNN model using the model.add functionality.

To define the model, iteratively call the appropriate layer function,
to define your mode. No changes to the training or infering process.


The training dataset is the MNIST handrwitten digit dataset, an immage classification
task.

The CNN will have an input layer of of (28,28,1), which is 28 pixels x 28 pixesl 
and greyscale, two convolution layers, maxpool, dropout, flatten, then a 128 nueron
full connected layer followed by a 10 neuron output layer using softmax as the
final activation, 
*/


/*Next we define the optimizer. Supported optimizers are any Keras provides. 
These are listed in this code's readme or on Keras.io.*/
optimizer := DATASET(
	[{'model', 'adadelt'},
	{'lr', '1.0'},
	{'momentum', '0.0'},
	{'decay', '0.0'},
	{'nesterov', 'false'},
	{'rho', '0.0'},
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
	{'epochs',12}]
, dt.parameterRec);


shape := [28,28,1];
nn := model.init();
nnInputLayer := model.add.Conv2D(nn,'relu', 32, [3, 3], shape);
nn2 := model.add.Conv2D(nnInputLayer, 'relu', 64, [3, 3]);
nn3 := model.add.MaxPooling2D(nn2, [2, 2]);
nn4 := model.add.dropout(nn3, 0.25);
nn5 := model.add.flatten(nn4);
nn6 := model.add.dense(nn5, 'relu', 128);
nn7 := model.add.dropout(nn6, 0.5);
nnOutputLayer := model.add.dense(nn7, 'softmax', 10);

mnistTrain := Datasets.mnist.LoadTrainData(shape);
mnistTest := Datasets.mnist.LoadTestData(shape);


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
