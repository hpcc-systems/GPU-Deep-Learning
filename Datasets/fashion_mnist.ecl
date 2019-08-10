/*
Fasion MNIST is in same format as MNSIT. Thus, the same methods are used, but exposed
under a new module for ease of usabiliyt and readability.
*/
IMPORT Python3;
IMPORT Datasets.data_types as dt;
IMPORT Datasets.mnist as m;

trainFM := DISTRIBUTE(DATASET('~fashionmnist::train', dt.mnist_data_type, THOR));
testFM := DISTRIBUTE(DATASET('~fashionmnist::test', dt.mnist_data_type, THOR));

train := global(trainFM,many);
test := global(testFM,few);


EXPORT fashion_mnist := MODULE
	EXPORT LoadData() := m.makeAllNumpy(train, test); //all data
	EXPORT LoadTrainData() := m.makeNumpy(train);
	EXPORT LoadTestData() := m.makeNumpy(test);
END;



