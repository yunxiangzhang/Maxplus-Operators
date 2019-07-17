# Maxplus block
Train, prune and test a morphological neural network.

This is a companion code to the paper ```Max-plus Operators Applied to Filter Selection and Model Pruning in Neural Networks``` by Yunxiang Zhang, Samy Blusseau, Santiago Velasco-Forero, Isabelle Bloch and Jesús Angulo, published in the International Symposium on Mathematical Morphology and Its Applications to Signal and Image Processing (ISMM) 2019.
https://link.springer.com/chapter/10.1007/978-3-030-20867-7_24
https://arxiv.org/pdf/1903.08072.pdf

## Example Usage
Provided TensorFlow and Keras are installed, the code should be run by the command
```
python testMaxPlus.py
```
The script testMaxPlus.py defines a morphological neural network composed of an input layer, a linear layer and a max-plus layer (or dilation layer), trains it on the MNIST or Fashion-MNIST dataset, then removes the unused filters from the linear layer and finally tests the reduced network on the test set.
