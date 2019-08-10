#option('outputLimit',2000);
IMPORT Datasets;
IMPORT Datasets.data_types as dt;
IMPORT Python3;


/*This is used to get the row counts to double check the python logic in the dataset loading is correct.*/
STRING getRowCount(DATASET(dt.np) inputData) := EMBED(Python3)

import pickle
import numpy as np

d = [n for n in inputData]

x = pickle.loads(bytes(d[0].x))
y = pickle.loads(bytes(d[0].y))

return str(x.shape) + str(y.shape)
ENDEMBED;



/*
Testing of the MNIST dataset
*/
mnistTrain := Datasets.mnist.LoadTrainData();
mnistTest := Datasets.mnist.LoadTestData();
//OUTPUT(getRowCount(mnistTrain));
//OUTPUT(getRowCount(mnistTest));

/*
Testing of the Fashion MNIST dataset.
*/
fmnistTrain := Datasets.fashion_mnist.LoadTrainData();
fmnistTest := Datasets.fashion_mnist.LoadTestData();
OUTPUT(getRowCount(fmnistTrain));
OUTPUT(getRowCount(fmnistTest));


