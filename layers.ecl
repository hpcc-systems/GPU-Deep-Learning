IMPORT Python3;

/*
This file serves as a wrapper around Keras layers to build NN with keras as a backend. 
All Keras layers can be found here: https://keras.io/layers/about-keras-layers/

This module "layers" outputs each layer, with appropriate inputs.

INIT() is required to initilize the model. Any layer is then added sequentially
to achieve the desired NN architecture. 

INPUT: All members accept a NN model to build upon, except INIT(), which accpets nothing
OUTPUT: All members output the given NN model with one more layer added

*/

//////////////////////////////Init//////////////////////////////////////////
STRING init() := EMBED(Python3)
import keras as k
import pickle

model = k.models.Sequential()

#model_config = model.get_config()
#return (pickle.dumps(model_config, 0), pickle.dumps(model_weights, 0), '', timer0)

#return bytearray(pickle.dumps(model.get_config(), 0))
return model.to_json()
ENDEMBED;

////////////////////////////Add Layer///////////////////////////////////////
STRING addLayer(STRING modelIn) := EMBED(Python3)
import keras as k
import pickle

#model = pickle.loads(modelIn)
model = k.models.model_from_json(modelIn)
model.add(k.layers.Dense(512, activation='relu', input_shape=(784,)))

return model.to_json()
#return bytearray(pickle.dumps(model.get_config(), 0))
ENDEMBED;

STRING test(STRING modelIn) := EMBED(Python3)
import keras as k
from keras.layers import Dense, Dropout
import pickle

#model = pickle.loads(modelIn)
model = k.models.model_from_json(modelIn)

model.add(Dropout(0.2))
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

return model.to_json()
#return bytearray(pickle.dumps(model.get_config(), 0))
ENDEMBED;



STRING addDense(STRING modelIn, SET of INTEGER shape, STRING activation, INTEGER neurons) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)
if shape: model.add(k.layers.Dense(neurons, activation=activation, input_shape=shape))
if not shape: model.add(k.layers.Dense(neurons, activation=activation))
return model.to_json()
ENDEMBED;


STRING addDropout(STRING modelIn, REAL d) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)
model.add(k.layers.Dropout(d))
return model.to_json()
ENDEMBED;


STRING addFlatten(STRING modelIn) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)
model.add(k.layers.Flatten())
return model.to_json()
ENDEMBED;


STRING addCONV2D(STRING modelIn, STRING activation, INTEGER neurons, SET of INTEGER filter, SET of INTEGER shape) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)

if shape: model.add(k.layers.Conv2D(neurons, filter, activation=activation, input_shape=shape))
if not shape: model.add(k.layers.Conv2D(neurons, filter, activation=activation))

return model.to_json()
ENDEMBED;


STRING addCONV1D(STRING modelIn, INTEGER neurons, INTEGER filter, SET of INTEGER shape, STRING activation) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)

if shape: model.add(k.layers.Conv1D(neurons, filter, activation=activation, input_shape=shape))
if not shape: model.add(k.layers.Conv1D(neurons, filter, activation=activation))

return model.to_json()
ENDEMBED;


STRING addMaxPooling2D(STRING modelIn, SET of INTEGER pool) := EMBED(Python3)
import keras as k
import pickle
model = k.models.model_from_json(modelIn)
model.add(k.layers.MaxPooling2D(pool_size=pool))
return model.to_json()
ENDEMBED;



EXPORT layers := MODULE
	EXPORT init() := init();
	EXPORT dense(STRING modelIn, STRING activation='', INTEGER neurons=0, SET of INTEGER shape=[]) := addDense(modelIn, shape, activation, neurons);
	EXPORT dropout(STRING modelIn, REAL d=0.0) := addDropout(modelIn, d);
	EXPORT flatten(STRING modelIn) := addFlatten(modelIn);
	
	EXPORT Conv2D(STRING modelIn, STRING activation='', INTEGER neurons=0, SET of INTEGER filter=[], SET of INTEGER shape=[]) := addCONV2D(modelIn, activation, neurons, filter, shape);
	EXPORT Conv1D(STRING modelIn, STRING activation='', INTEGER neurons=0, INTEGER filter=0, SET of INTEGER shape=[]) := addCONV1D(modelIn, neurons, filter, shape, activation);
	EXPORT MaxPooling2D(STRING modelIn, SET of INTEGER pool=[]) := addMaxPooling2D(modelIn, pool);

END;
