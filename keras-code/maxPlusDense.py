'''
Classes and function used to train and prune a max-plus block
By Santiago Velasco-Forero & Samy Blusseau

It is a companion code to the paper 

Max-plus Operators Applied to Filter Selection and Model Pruning in Neural Networks

authored by Yunxiang Zhang, Samy Blusseau, Santiago Velasco-Forero, Isabelle Bloch and Jesús Angulo,
published in the International Symposium on Mathematical Morphology and Its Applications to Signal and Image Processing (ISMM) 2019.
https://link.springer.com/chapter/10.1007/978-3-030-20867-7_24
https://arxiv.org/pdf/1903.08072.pdf
'''


import numpy as np
import keras
from keras.layers import Dense
import keras.backend as K
from keras.layers import GaussianNoise
from keras.layers import Dropout, Add, concatenate
from keras.optimizers import Adam
from keras.layers.core import Activation
from keras import initializers, regularizers, constraints
from keras.engine.topology import Layer, InputSpec
from keras.models import Input, Model

#-----------------------------------------------------------------------#
class MaxPlusDense(Layer):
    """A MaxPlus layer. TESTING MODE
    A `MaxPlus` layer takes the (element-wise + Bias) maximum of
      # Arguments
        output_dim: int > 0.
        init: name of initialization function for the weights of the layer
            (see [initializations](../initializations.md)),
            or alternatively, Theano function to use for weights
            initialization. This parameter is only relevant
            if you don't pass a `weights` argument.
        weights: list of Numpy arrays to set as initial weights.
            The list should have 2 elements, of shape `(input_dim, output_dim)`
            and (output_dim,) for weights and biases respectively.
        b_regularizer: instance of [WeightRegularizer](../regularizers.md),
            applied to the bias.
        activity_regularizer: instance of [ActivityRegularizer](../regularizers.md),
            applied to the network output.
        b_constraint: instance of the [constraints](../constraints.md) module,
            applied to the bias.
        bias: whether to include a bias
            (i.e. make the layer affine rather than linear).
        input_dim: dimensionality of the input (integer). This argument
            (or alternatively, the keyword argument `input_shape`)
            is required when using this layer as the first layer in a model.
    # Input shape
        2D tensor with shape: `(nb_samples, input_dim)`.
    # Output shape
        2D tensor with shape: `(nb_samples, output_dim)`.
    """

    def __init__(self, output_dim,
                 nb_feature=1,
                 init='ones',
                 weights=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        self.output_dim = output_dim
        self.init = initializers.get(init)

        self.b_regularizer = regularizers.get(b_regularizer)
        self.activity_regularizer = regularizers.get(activity_regularizer)

        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)
        self.input_dim = input_dim
        if self.input_dim:
            kwargs['input_shape'] = (self.input_dim,)
        super(MaxPlusDense, self).__init__(**kwargs)

    def build(self, input_shape):
        input_dim = input_shape[1]
        self.nb_feature = input_dim #Dilate Layer Always has Identity Matrix on W
        self.input_spec = InputSpec(dtype=K.floatx(),shape=(None, input_dim))
        if self.bias:
            self.b = self.add_weight((self.nb_feature, self.output_dim,),
                                     initializer='zero',
                                     name='b',
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return (input_shape[0], self.output_dim)

    def call(self, x):
        output=K.concatenate([K.reshape(x, [-1, self.nb_feature, 1])]*self.output_dim)
        output +=self.b
        output = K.max(output, axis=1)
        return output

    def get_config(self):
        config = {'output_dim': self.output_dim,
                  'init': initializers.serialize(self.init),
                  'nb_feature': self.nb_feature,
                  'b_regularizer': regularizers.serialize(self.b_regularizer),
                  'activity_regularizer': regularizers.serialize(self.activity_regularizer),
                  'b_constraint': constraints.serialize(self.b_constraint),
                  'bias': self.bias,
                  'input_dim': self.input_dim}
        base_config = super(MaxoutDense, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#    

class ArgMaxPlusDense(MaxPlusDense):
    '''class ArgMaxPlusDense(MaxPlusDense) Inherits from class
    MaxPlusDense Computes the argmax(input + bias) instead of the max
    This allows to figure out the filters in the previous layer whose
    responses achieve the maximum in the MaxDense layer. Everything
    else is unchanged compared to the maxPlusDense layer

    '''
    
    def __init__(self, output_dim,
                 nb_feature=1,
                 init='ones',
                 weights=None,
                 b_regularizer=None,
                 activity_regularizer=None,
                 b_constraint=None,
                 bias=True,
                 input_dim=None,
                 **kwargs):
        
        super().__init__(output_dim,
                         nb_feature=1,
                         init='ones',
                         weights=None,
                         b_regularizer=None,
                         activity_regularizer=None,
                         b_constraint=None,
                         bias=True,
                        input_dim=None,
                         **kwargs)
        
    def call(self, x):
        output=K.concatenate([K.reshape(x, [-1, self.nb_feature, 1])]*self.output_dim)
        output +=self.b
        output = K.argmax(output, axis=1)
        return output
    
#-----------------------------------------------------------------------#
#-----------------------------------------------------------------------#

def getMaxPlusBlock(input_dim, proj_dim, num_classes=10, dropout_rate=.5):
    '''getMaxPlusBlock(input_dim, proj_dim, num_classes=10, dropout_rate=.5)

    Builds and returns a Max-plus block model with input layer of size
    input_dim, a linear layer of size proj_dim and an output layer of
    size num_classes

    Input:

    input_dim: dimension of the inut vector (number of neurons in the input layer)
    proj_dim: number of neurons in the linear layer
    num_classes: number of classes, which also the output dimension
    dropoout_rate: proportion of units dropped during training in the linear layer

    Output:

    The max-plus block model

    '''
    xin = Input(shape=(input_dim,))
    xin1 = GaussianNoise(.2)(xin)
    x1=Dense(proj_dim, activation='relu',name='linear',use_bias=False)(xin1)
    added = Dropout(dropout_rate, name='dropout_linear')(x1)
    MPlus = MaxPlusDense(num_classes,name='MaxPlus')(added)
    y=Activation('softmax')(MPlus)
    return Model(xin, y)

def getArgmaxPlusBlock(input_dim, proj_dim, num_classes=10):
    '''getArgmaxPlusBlock(input_dim, proj_dim, num_classes=10)
    
    Builds and returns an argmax-plus block model with input layer of
    size input_dim, a linear layer of size proj_dim and an output
    layer of size num_classes

    Input:

    input_dim: dimension of the inut vector (number of neurons in the input layer)
    proj_dim: number of neurons in the linear layer
    num_classes: number of classes, which also the output dimension

    Output:

    The argmax-plus block model

    '''
    xin = Input(shape=(input_dim,))
    xin1 = GaussianNoise(.2)(xin)
    x1=Dense(proj_dim, activation='relu',name='linear',use_bias=False)(xin1)
    argMPlus = ArgMaxPlusDense(num_classes,name='argMaxPlus')(x1)
    return Model(xin, argMPlus)

def findActiveFilters(maxPlusModel, x_set, y_set, ths_rate=.005):
    '''findActiveFilters(maxPlusModel, x_set, y_set, ths_rate=.005)

    Determines which filters (neurons) are active in the linear
    layer of the max-plus block maxPlusModel.

    An active filter is defined as a filter whose response achieves
    the maximum of a max-plus neuron for at least 100*ths_rate % of
    the samples from x_set (default is 0.5%). If ths_rate is set to 0,
    then a filter is considered active as soon as it achieves the
    maximum at least once among the samples.

    Input:

    maxPlusModel: A max-plus block with input and output dimensions consistent to x_set and y_set
    x_set: a numpy array of size nsamples x input_dimension; it represents a set to which the input model can be applied
    y_set: a numpy array of size nsamples x output_dimension; it represents the ground truth associated to x_set 
    ths_rate: a number between 0 and 1 that modulates the definition of active filter; the closer to zero, the less active a filter needs to be in order to be selected

    Output: 

    all_fiters_u: np array containing the unique indexes of the active filters
    all_filters_pc: list of num_class lists, each one containing the active 
    filters of the corresponding class

    '''
    # Build the argMaxPlus model corresponding to the input
    # maxPlusModel, by copying its weights
    W1=maxPlusModel.get_layer('linear').get_weights()
    input_dim = W1[0].shape[0]
    proj_dim = W1[0].shape[1]
    W2=maxPlusModel.get_layer('MaxPlus').get_weights()
    num_classes = W2[0].shape[1]
    model2=getArgmaxPlusBlock(input_dim, proj_dim, num_classes)
    model2.get_layer('linear').set_weights(W1)
    model2.get_layer('argMaxPlus').set_weights(W2)
    
    # Get the indexes of the filters achieving the maxima in the max-plus
    # layer, for all the input samples x_set
    pred_argmax = model2.predict(x_set)

    # Select the active filters
    all_filters = list()
    all_filters_pc = list()
    for class_num in range(num_classes):
        idxs_class = np.where(y_set[:, class_num] == 1)
        n_members = len(idxs_class[0])
        filters_activity = pred_argmax[idxs_class[0], class_num]
        histo, be = np.histogram(filters_activity, range(proj_dim+1))
        int_ths = int(ths_rate*n_members)
        used_filters = np.where(histo > int_ths)
        histo = histo[used_filters[:]]
        idx = np.argsort(histo)[::-1]
        all_filters.extend(used_filters[0][idx])
        all_filters_pc.append(list(used_filters[0][idx].tolist()))
    # Convert list of filters into numpy array
    all_filters = np.array(all_filters)
    # Remove redundant indexes
    all_filters_u = np.unique(all_filters)
    return all_filters_u, all_filters_pc


def removeInactiveFilters(inputModel, idxsFiltersToKeep, dropout_rate=.5, same_dropout = False):
    '''removeInactiveFilters(inputModel, idxsFiltersToKeep, dropout_rate=.5, same_dropout = False)

    Creates a reduced max-plus block from the input one, by removing
    the linear filters which are not in the list idxsFiltersToKeep

    If one wants to train the reduced model, the dropout rate can be set to the desired value dropout_rate 
    or the same rate as the input model can be re-used, by setting same_dropout = True

    Input: 

    inputModel: A max-plus block model
    idxsFiltersToKeep: np array containing the indexes of the filters to keep from the input model's linear layer
    dropout_rate: a number between 0 and 1, proportion of dropped units in the linear layer during training; ignored if same_dropout is set to True
    same_dropout: boolean defining whether the same dropout rate as the input model should be kept; If True, dropout_rate is ignored

    Output:
    
    The pruned max-plus block

    '''
    idx = idxsFiltersToKeep
    W1 = inputModel.get_layer('linear').get_weights()
    input_dim = W1[0].shape[0]
    W1Red=W1
    W1Red[0]=W1Red[0][:,idx]
    W2 = inputModel.get_layer('MaxPlus').get_weights()
    num_classes = W2[0].shape[1]
    W2red=W2
    W2red[0]=W2red[0][idx,:]
    nLinearFilters = idx.shape[0]
    if same_dropout:
        dpout_rate = inputModel.get_layer('dropout_linear').get_config()['rate']
    else:
        dpout_rate = dropout_rate
    #Building Reduced Model
    reducedModel = getMaxPlusBlock(input_dim, nLinearFilters, num_classes, dpout_rate)
    reducedModel.get_layer('linear').set_weights(W1Red)
    reducedModel.get_layer('MaxPlus').set_weights(W2red)
    return reducedModel


def ReduceModel(inputModel, x_set, y_set, dropout_rate=.5, same_dropout = False, ths_rate = .005):
    '''ReduceModel(inputModel, x_set, y_set, dropout_rate=.5, same_dropout = False, ths_rate = .005)

    Finds the active filters in the linear layer of the max-plus block
    inputModel, by analysing its behaviour on x_set

    Then removes the inactive filters to return a reduced max-plus block

    Input: 

    inputModel: A max-plus block with input and output dimensions consistent to x_set and y_set
    x_set: a numpy array of size nsamples x input_dimension; it represents a set to which the input model can be applied
    y_set: a numpy array of size nsamples x output_dimension; it represents the ground truth associated to x_set
    dropout_rate: a number between 0 and 1, proportion of dropped units in the linear layer during training; ignored if same_dropout is set to True
    same_dropout: boolean defining whether the same dropout rate as the input model should be kept; If True, dropout_rate is ignored
    ths_rate: a number between 0 and 1 that modulates the definition of active filter; the closer to zero, the less active a filter needs to be in order to be selected

    Output:

    reducedModel: the pruned max-plus block
    nfilt: the number of filters in the linear layer of the pruned model
    activeFilters: np array containing the unique indexes of the active filters
    activeFiltersPc: list of num_class lists, each one containing the active 
    filters of the corresponding class

    '''
    activeFilters, activeFiltersPc = findActiveFilters(inputModel,x_set, y_set, ths_rate)
    nfilt = activeFilters.shape[0]
    reducedModel = removeInactiveFilters(inputModel, activeFilters, dropout_rate, same_dropout)
    return reducedModel, nfilt, activeFilters, activeFiltersPc
