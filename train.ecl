/*
These methods are imported by the model file so one can consume these in the form of "model.train()"
instead of train.train().
*/

IMPORT Python3;
IMPORT Datasets;
IMPORT Datasets.data_types as dt;

/*default optimizer
optimizer := DATASET(
[{'model', 'rmsprop'},
{'lr', '0.01'},
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

trainingParameters := DATASET([{'batch_size',128}, {'epochs',20}], dt.parameterRec);
*/


dt.trainedModel buildNN(STRING modelIn, DATASET(dt.optimizerRec) optimizerConfig, DATASET(dt.parameterRec) trainingConfig, DATASET(dt.np) inputData) := EMBED(Python3)
import keras as k
import pickle
from keras import optimizers as optimizers
from keras import backend as BK
BK.clear_session()

opti = {k.lower():v.lower() for (k,v) in optimizerConfig}
config = {k.lower():v for (k,v) in trainingConfig}

batch_size = config.get('batch_size', 128)
epochs = config.get('epochs', 5)
loss = opti.get('loss_function', 'categorical_crossentropy')

defaultOptimizer = optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)

dat = [n for n in inputData]
x = pickle.loads(bytes(dat[0].x))
y = pickle.loads(bytes(dat[0].y))



if opti['model'] == 'sgd':
	if opti['nesterov'] == 'false': nesterov=False
	else: nesterov=True
	optimizer = optimizers.SGD(lr=float(opti['lr']), momentum=float(opti['momentum']), decay=float(opti['decay']), nesterov=nesterov)

elif opti['model'] == 'rmsprop':
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.RMSprop(lr=float(opti['lr']), rho=float(opti['rho']), epsilon=epsilon, decay=float(opti['decay']))

elif opti['model'] == 'adagrad':
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.Adagrad(lr=float(opti['lr']), epsilon=epsilon, decay=float(opti['decay']))

elif opti['model'] == 'adadelt':
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.Adadelta(lr=float(opti['lr']), rho=float(opti['rho']), epsilon=epsilon, decay=float(opti['decay']))

elif opti['model'] == 'adam':
	if opti['amsgrad'] == 'false': amsgrad=False
	else: amsgrad=True
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.Adam(lr=float(opti['lr']), beta_1=float(opti['beta_1']), beta_2=float(opti['beta_2']), epsilon=epsilon, decay=float(opti['decay']), amsgrad=amsgrad)

elif opti['model'] == 'adamax':
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.Adamax(lr=float(opti['lr']), beta_1=float(opti['beta_1']), beta_2=float(opti['beta_2']), epsilon=epsilon, decay=float(opti['decay']))

elif opti['model'] == 'nadam':
	if opti['epsilon'] == 'none': epsilon=None
	else: epsilon=float(opti['epsilon'])
	optimizer = optimizers.Nadam(lr=float(opti['lr']), beta_1=float(opti['beta_1']), beta_2=float(opti['beta_2']), epsilon=epsilon, schedule_decay=float(opti['schedule_decay']))

else:
	optimizer = defaultOptimizer

#defaults of the optimizers
#optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False)
#optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=None, decay=0.0)
#optimizers.Adagrad(lr=0.01, epsilon=None, decay=0.0)
#optimizers.Adadelta(lr=1.0, rho=0.95, epsilon=None, decay=0.0)
#optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
#optimizers.Adamax(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0)
#optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)

model = k.models.model_from_json(modelIn)

model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])


#history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=0,validation_data=(x_test, y_test))
#this is the beginning of the trainAndTest specific code
#score = model.evaluate(x_test, y_test, verbose=0)
#return 'Loss: ' + str(score[0]) + ' | Accuracy: ' + str(score[1]) + ' | Shapes: ' + str(x_train.shape) + str(y_train.shape) + str(x_test.shape) + str(y_test.shape) + ' | Epochs: ' + str(epochs)

history = model.fit(x, y,batch_size=batch_size,epochs=epochs,verbose=0,validation_split=0.2)
byteOut = bytearray(pickle.dumps(model,0))
	
#['acc', 'loss', 'val_acc', 'val_loss']
results = 'Accuracy: ' + str(history.history['acc'][-1]) + ' | Loss: ' + str(history.history['loss'][-1]) + ' | Validation Accuracy: ' + str(history.history['val_acc'][-1]) + ' | Validation Loss: ' + str(history.history['val_loss'][-1])
return (byteOut, results)
ENDEMBED;

//Test the NN in a keras/pythonic way
//returns a performance metric
STRING testNN(dt.trainedModel modelIn, DATASET(dt.np) inputData) := EMBED(Python3)
import keras as k

import pickle
from keras import optimizers as optimizers

dat = [n for n in inputData]
#x_train = pickle.loads(bytes(dat[0].x_train))
#y_train = pickle.loads(bytes(dat[0].y_train))
x = pickle.loads(bytes(dat[0].x))
y = pickle.loads(bytes(dat[0].y))

model = pickle.loads(modelIn.model)

score = model.evaluate(x, y, verbose=0)
return 'Test Loss: ' + str(score[0]) + ' | Test Accuracy: ' + str(score[1]) + ' | Test Shapes: '  + str(x.shape) + str(y.shape)
ENDEMBED;


//return dataset of predictions
//accepts a dataset of one or more instances to predict on
//returns a dataset of onehot encoded predictions. each one hot is a set of ints
DATASET(dt.oneHot) predictClass(dt.trainedModel modelIn, DATASET(dt.mnist_data_type) inputData) := EMBED(Python3)
import keras as k
import pickle
import numpy as np

trainData = [[n.label, np.asarray(n.image, dtype='B')] for n in inputData] #list of lists
x = np.array([np.array(xi[1]) for xi in trainData])
y = np.array([np.array(xi[0]) for xi in trainData])

model = pickle.loads(modelIn.model)

#predict(x, batch_size=64, verbose=0)

predictions = model.predict(x).astype(int)
#return [[1,2,3,4],[5,6,7,8]]
return predictions.tolist()
ENDEMBED;

//these are imported under "model" so you can call them like "model.train()"
EXPORT train := MODULE
	EXPORT train(STRING modelIn, DATASET(dt.optimizerRec) optimizerConfig, DATASET(dt.parameterRec) trainingConfig, DATASET(dt.np) inputData) 
			:= buildNN(modelIn, optimizerConfig, trainingConfig, inputData);
	
	EXPORT test(dt.trainedModel modelIn, DATASET(dt.np) inputData) 
			:= testNN(modelIn, inputData); 
	//Predict takes the dataset, in HPCC format, and makes predictions
	Export predict(dt.trainedModel modelIn, DATASET(dt.mnist_data_type) inputData) := predictClass(modelIn, inputData);
END;

