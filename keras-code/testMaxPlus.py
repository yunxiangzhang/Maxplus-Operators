'''
Short tutorial to train and prune max-plus blocks on Mnist and Fashion mnist datasets
By Santiago Velasco-Forero & Samy Blusseau

It is a companion code to the paper 

Max-plus Operators Applied to Filter Selection and Model Pruning in Neural Networks

authored by Yunxiang Zhang, Samy Blusseau, Santiago Velasco-Forero, Isabelle Bloch and Jesús Angulo,
published in the International Symposium on Mathematical Morphology and Its Applications to Signal and Image Processing (ISMM) 2019
https://link.springer.com/chapter/10.1007/978-3-030-20867-7_24
https://arxiv.org/pdf/1903.08072.pdf

'''

from maxPlusDense import *
from keras.datasets import mnist,fashion_mnist

input_dim=784 # Corresponds to the num. of pixels of each input image; do not change
proj_dim= 64  # Number of filters in the linear layer
dropout_rate=0.5
ths_rate = 0 # Parameter that modulates the definition of active filter
                # Should be between 0 and 1
                # The closer to zero, the less active a filter needs to be in order to be selected
                # See the function findActiveFilters in maxPlusDense.py for more details

batch_size = 128
epochs = 10
num_classes= 10 # Number of classes in the dataset; do not change


# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data() # MNIST dataset
#(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data() # Fashion MNIST dataset https://github.com/zalandoresearch/fashion-mnist

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Build, train and evaluate the initial max-plus block
model = getMaxPlusBlock(input_dim,proj_dim,num_classes,dropout_rate)
model.compile(loss='categorical_crossentropy',optimizer=Adam(),metrics=['accuracy'])
history = model.fit(x_train, y_train,batch_size=batch_size,epochs=epochs,verbose=1,validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('\n')
print('Initial MaxPlus Block')
print('Number of linear filters: ', proj_dim, '; Number of Parameters: ', model.count_params())
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Choose the sample set used to define the active filters
keepOnlyWellClassified = True
if keepOnlyWellClassified:
    # keep only the well classified samples of the training set to define the active filters
    pred = model.predict(x_train)
    correctly_classified = np.where(sum((pred*y_train).T)>=np.amax(pred, axis=1))
    print('Train accuracy:', len(correctly_classified[0])/60000)
    x_set = x_train[correctly_classified[0],:]
    y_set = y_train[correctly_classified[0],:]
else:
    # keep all the samples of the training set to define the active filters
    x_set = x_train
    y_set = y_train

# Build the pruned model
reducedModel, nfilt, activeFilters, activeFiltersPc = ReduceModel(model, x_set, y_set, dropout_rate, ths_rate = ths_rate)
reducedModel.compile(loss='categorical_crossentropy', optimizer=Adam(), metrics=['accuracy'])
# Evaluate the pruned model
score = reducedModel.evaluate(x_test, y_test, verbose=0)
print('\n')
print('Pruned MaxPlus Block')
print('Number of linear filters: ', nfilt, '; Number of Parameters: ', reducedModel.count_params())
print('Test loss:', score[0])
print('Test accuracy:', score[1])

print('\n')
print(activeFiltersPc)
