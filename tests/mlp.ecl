#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This test code generates a MLP model and trains and tests it on the given data.

hiddenLayers = length of array is number of hidden layers, value is number of neurons
in each hidden layer. Dropout of 0.2 after each layer

activation = first activation is the actiation for all but the output layers,
second activation is the output layer's activation function

IOShape = the input layer shape (features) and output layer shape (classes)

Optimizer = is the optimizer used and all of its parameters, see https://keras.io/optimizers/
for a list of all models and hyperparameters.
All values are represented as strings and then converted into intgers, floats, booleans or None types

Training Parameters = bach size and epochs to train for

mnistData = is the dataset used for building the NN

*/

hiddenLayers := [512,512]; //2 hidden layers, each with 512 neurons
activation := ['relu','softmax'];
IOShape := [784, 10]; //784 input, 10 output
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

trainingParameters := DATASET(
	[{'batch_size',128},
	{'epochs',20}]
, dt.parameterRec);

mnistData := Datasets.fashion_mnist.LoadData();
mnistTrain := Datasets.fashion_mnist.LoadTrainData();
mnistTest := Datasets.fashion_mnist.LoadTestData();

//makes a NN architecture
NN := model.mlp(hiddenLayers, activation, IOShape);
//Passes the NN architecture into this.train to train and test
trainedModel := model.train(NN, optimizer, trainingParameters, mnistTrain);// : Persist('nn_train_dev');
testPerformanceMetrics := model.test(trainedModel, mnistTest);

testDS0 := CHOOSEN(DATASET('~mnist::test', dt.mnist_data_type, THOR), 10);
testDS := global(testDS0,FEW);
predictResults := model.predict(trainedModel, testDS);

OUTPUT(trainedModel.performanceMetrics);
OUTPUT(testPerformanceMetrics); //this outputs test metrics
OUTPUT(predictResults);


