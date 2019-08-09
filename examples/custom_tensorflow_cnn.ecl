#option('outputLimit',2000);
IMPORT Python3;
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT model as model;
IMPORT train;

/*
This demonstrates how you can write custom TensorFlow/Keras code within
the HPCC Systems Platform and this bundle. We will use Keras, but TensorFlow
code will work the same.

We will use the MNIST dataset for our example. And build a CNN.

The customKeras() function does not necessarily need any inputs.

This is expected to achieve a ~99% accuracy after 12 epochs.
*/

STRING customKeras() := EMBED(Python3)
import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

#############################################
# Start your custom keras modeling here
#############################################
batch_size = 128
num_classes = 10
epochs = 12
input_shape = (28,28,1)

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

#############################################
# Return the model via Keras' to_json()
# the model architecture you design will then
# be compiled, and trained in the next steps
#############################################
return model.to_json()
ENDEMBED;

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
	{'epochs',12}]
, dt.parameterRec);


//Define your model in python
NN := customKeras();


/*
LoadTrainData() and LoadTestData() both accept a shape. This is represented
as an array of integers in ecl, and is a tuple in python. It reshapes the
data with numpy.reshape(). Omitting the parameter produces no reshaping.

Test and Train need to have same shape.
*/
shape := [28,28,1];
mnistTrain := Datasets.mnist.LoadTrainData(shape);
mnistTest := Datasets.mnist.LoadTestData(shape);


/*
Once the model architecture is defined, and the data is loaded and correctly
shaped, training, testing, and predicting can then be acomplished as shown
below.
*/
trainedModel := model.train(NN, optimizer, trainingParameters, mnistTrain);
testPerformanceMetrics := model.test(trainedModel, mnistTest);
OUTPUT(trainedModel.performanceMetrics);
OUTPUT(testPerformanceMetrics);



