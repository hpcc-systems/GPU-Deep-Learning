#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

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


//nn := model.init();
//nn1 := model.add(nn);
//nn2 := model.add1(nn1);

nn := model.init();
//nn1 := model.add.dense(nn, 'relu', 512, [784]);
//nn2 := model.add.dropout(nn1,0.2);
//nn3 := model.add.dropout(nn2);

nn1 := model.add.Conv2D(nn, 'relu', 32, [3,3]);
nn2 := model.add.MaxPooling2D(nn, [2,2]);
OUTPUT(nn2);

mnistTrain := Datasets.mnist.LoadTrainData();
mnistTest := Datasets.mnist.LoadTestData();


/*
Now we can start the actual training. The first line will train the model
and the second line will take the trained model and test its ability
to generalize by testing on unseen test data.

Both of these output performance metrics, first the training and validation,
second is the test performance metrics.
*/
//trainedModel := model.train(nn, optimizer, trainingParameters, mnistTrain);
//testPerformanceMetrics := model.test(trainedModel, mnistTest);
//OUTPUT(trainedModel.performanceMetrics);
//OUTPUT(testPerformanceMetrics);



//EXPORT add_layers := 'todo';