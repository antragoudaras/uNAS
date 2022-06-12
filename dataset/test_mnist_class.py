from mnist import MNIST

dataset_mnist = MNIST()
train_data = dataset_mnist.train_dataset()
val_data = dataset_mnist.validation_dataset()
test_data = dataset_mnist.test_dataset()
print('Dummy print')