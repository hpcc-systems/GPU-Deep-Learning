#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This example is the same as the MNIST, but using the similar Fashion MNIST dataset.

Expected testing accuracy is ~89.3%
*/

hiddenLayers := [512,512];
activation := ['relu','softmax'];
IOShape := [784, 10];
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


NN := model.mlp(hiddenLayers, activation, IOShape);
fmnistTrain := Datasets.fashion_mnist.LoadTrainData();
fmnistTest := Datasets.fashion_mnist.LoadTestData();


trainedModel := model.train(NN, optimizer, trainingParameters, fmnistTrain);
testPerformanceMetrics := model.test(trainedModel, fmnistTest);
OUTPUT(trainedModel.performanceMetrics);
OUTPUT(testPerformanceMetrics);

testDS0 := CHOOSEN(DATASET('~fashionmnist::test', dt.mnist_data_type, THOR), 10);
testDS := global(testDS0,FEW);
predictResults := model.predict(trainedModel, testDS);
OUTPUT(predictResults);
