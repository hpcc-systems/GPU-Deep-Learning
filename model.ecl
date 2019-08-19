IMPORT Python3;
IMPORT train as tr;
IMPORT Datasets.data_types as dt;
IMPORT layers as L;


/*
ECL Model module should return the variou different types of modles.

Perhaps have it be a class (is this possible in ecl) that can sequentially add layers

*/
/*
input parameters
the model shape, for MLP, will be an array of int, length is # of layers, each int is num of neurons

Thoughts:
have shape and actiation be arrays, and have it adapt to the length that is inputed

python imports via input parameters?
*/
//shape := [128,128,128];
//activation := ['relu','softmax', 'relu'];
//dataShape := [784, 10];
//OUTPUT(shape);

STRING createMLP(
	SET of INTEGER shape=[128], 
	SET of STRING activations=['relu','softmax'], 
	SET of INTEGER IOshape
) 
:= EMBED(Python3)

import keras as k

num_classes = IOshape[-1]
activation = activations[0]
finalActivation = activations[-1]
numFeatures = IOshape[0]

model = k.models.Sequential()

for i, neurons in enumerate(shape):
	if i == 0: #first layer
		model.add(k.layers.Dense(neurons, activation=activation, input_shape=(numFeatures,)))
		model.add(k.layers.Dropout(0.2))
	if i != len(shape)-1: #hidden layers
		model.add(k.layers.Dense(neurons, activation=activation))
		model.add(k.layers.Dropout(0.2))
	if i == len(shape)-1:#output layer
		model.add(k.layers.Dense(num_classes, activation=finalActivation))

#with open('/home/ubuntu/temp/report.txt','w') as fh: 
#	# Pass the file handle in as a lambda function to make it callable 
#	model.summary(print_fn=lambda x: fh.write(x + '\n'))

return model.to_json()
#return str(finalActivation)
ENDEMBED;


EXPORT model := MODULE
	EXPORT init() := L.init();
	EXPORT add := L;
	
	/*These are used to create an NN of certain categories, in the desired architecture.
		ex: MLP() will create a fully connected NN with the desired number of hidden layers and neurons*/
	EXPORT mlp(SET of INTEGER shape, SET of STRING activations, SET of INTEGER IOshape) := createMLP(shape, activations, IOshape);
	
	/*Train() builds and trains, in input training data, a NN. A NN can be created using THIS.mlp()*/
	EXPORT train(STRING modelIn, DATASET(dt.optimizerRec) optimizerConfig, DATASET(dt.parameterRec) trainingConfig, DATASET(dt.np) inputData) := tr.train(modelIn, optimizerConfig, trainingConfig, inputData);
	
	/*Test() tests the model, in a pythonic way, and outputs performance metrics, as opposed to actual predictions.*/
	EXPORT test(dt.trainedModel modelIn, DATASET(dt.np) inputData) := tr.test(modelIn, inputData); 
	
	/*Predict() takes the dataset, in HPCC format, and makes predictions
		outputs the results in one hot encoding with integers */
	Export predict(dt.trainedModel modelIn, DATASET(dt.mnist_data_type) inputData) := tr.predict(modelIn, inputData);
END;


