IMPORT Python3;
IMPORT Datasets.data_types as dt;

	
//train0 := DISTRIBUTE(CHOOSEN(DATASET('~mnist::train', dt.mnist_data_type, THOR), 60));
//test0 := DISTRIBUTE(CHOOSEN(DATASET('~mnist::test', dt.mnist_data_type, THOR), 10));
train0 := DISTRIBUTE(DATASET('~mnist::train', dt.mnist_data_type, THOR));
test0 := DISTRIBUTE(DATASET('~mnist::test', dt.mnist_data_type, THOR));
trainBig0 := DISTRIBUTE(DATASET('~mnist::big::train', dt.mnist_data_type, THOR));
testBig0 := DISTRIBUTE(DATASET('~mnist::big::test', dt.mnist_data_type, THOR));

train := global(train0,MANY);
test := global(test0,MANY);
trainBig := global(trainBig0,MANY);
testBig := global(testBig0,MANY);

DATASET(dt.np_ds_type) makeAllNumpy(DATASET(dt.mnist_data_type) trainInput, DATASET(dt.mnist_data_type) testInput) := EMBED(Python3 : globalscope('mnist'),persist('global'))
global x_train
global y_train
global x_test
global y_test
global num_classes
global model

import numpy as np
import pickle
from keras import utils as utils

#put the training into a variable as to avoid passing back and forth through HPCC


num_classes = 10

trainData = [[n.label, np.asarray(n.image, dtype='B')] for n in trainInput] #list of lists
x_train = np.array([np.array(xi[1]) for xi in trainData])
y_train = np.array([np.array(xi[0]) for xi in trainData])

testData = [[n.label, np.asarray(n.image, dtype='B')] for n in testInput]
x_test = np.array([np.array(xi[1]) for xi in testData])
y_test = np.array([np.array(xi[0]) for xi in testData])


#x_train shape = (60000, 784)
#x_test shape = (10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

#This one-hot-encodes the class (y) labels
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)


#output a bytearray for the HPCC "DATA" type

out1 = bytearray(pickle.dumps(x_train, 0))
out2 = bytearray(pickle.dumps(y_train, 0))
out3 = bytearray(pickle.dumps(x_test, 0))
out4 = bytearray(pickle.dumps(y_test, 0))
return [(out1, out2, out3, out4)]
#return [(out,'2','3','4')]
ENDEMBED;

DATASET(dt.np) makeNumpy(DATASET(dt.mnist_data_type) inputData, SET of INTEGER shape) := EMBED(Python3 : globalscope('mnist'),persist('global'))
global x
global y

import numpy as np
import pickle
from keras import utils as utils

#put the training into a variable as to avoid passing back and forth through HPCC

num_classes = 10

trainData = [[n.label, np.asarray(n.image, dtype='B')] for n in inputData] #list of lists
x = np.array([np.array(xi[1]) for xi in trainData])
y = np.array([np.array(xi[0]) for xi in trainData])

#x_train shape = (60000, 784)
if shape:
	len = (x.shape[0],)
	shape = tuple(shape)
	x = x.reshape(len+shape)

x = x.astype('float32')
x /= 255

#This one-hot-encodes the class (y) labels
y = utils.to_categorical(y, num_classes)

out1 = bytearray(pickle.dumps(x, 0))
out2 = bytearray(pickle.dumps(y, 0))

return [(out1, out2)]
#return str(y)
ENDEMBED;

//#option('outputLimit',2000);
//m := makeNumpy(test);
//OUTPUT(m);

//OUTPUT(COUNT(makeNumpy(trainBig, testBig)));

EXPORT mnist := MODULE
	EXPORT LoadData() := makeAllNumpy(train, test); //all data
	EXPORT LoadTrainData(SET of INTEGER shape=[]) := makeNumpy(train, shape);
	EXPORT LoadTestData(SET of INTEGER shape=[]) := makeNumpy(test, shape);
	EXPORT LoadBigData(INTEGER n) := makeAllNumpy(CHOOSEN(trainBig, n), CHOOSEN(testBig, n));
	
	/*For use in fasion mnist since it is in same format as mnist, just different data.*/
	EXPORT makeNumpy(DATASET(dt.mnist_data_type) train, SET of INTEGER shape=[]) := makeNumpy(train, shape);
	EXPORT makeAllNumpy(DATASET(dt.mnist_data_type) train, DATASET(dt.mnist_data_type) test) := makeAllNumpy(train, test);
END;
