IMPORT Datasets;
IMPORT Datasets.data_types as dt;

IMPORT model as model;

fashiontrain := DISTRIBUTE(CHOOSEN(DATASET('~fashionmnist::train', dt.mnist_data_type, THOR), 60));
//OUTPUT(fashiontrain);
train := DISTRIBUTE(CHOOSEN(DATASET('~fashionmnist::test', dt.mnist_data_type, THOR), 60));
//OUTPUT(train);


shape := [128,128,128];
activation := ['relu','softmax', 'relu'];
dataShape := [784, 10];

OUTPUT(model.mlp(shape, activation, datashape));

//OUTPUT(Datasets.mnist.LoadData());




//EXPORT datasets := 'todo';