from sklearn.datasets import make_classification

# set a class number as a variable to be used in the make_classification and in the models themselves. This could also be done for all of the other parameters used in the make_class function
class_num = 8

# put epochs in a variable since both models are using the same numer of epochs. It is easier to change a variable than hard code in multiple models.
epochs = 100

# Creating the date
X1, y1 = make_classification(n_samples=10000, n_features=17, n_informative=6, n_redundant=0, n_repeated=0, n_classes=class_num,
                             n_clusters_per_class=3, weights=None, flip_y=.3, class_sep=.4, hypercube=False, shift=3,
                             scale=2, shuffle=True, random_state=840780)

# Create the column names for the spreadsheet(csv file)
columns_labels = ['Learning_Rate', 'n_features', 'n_informative', 'n_classes', 'flip_y', 'class_sep', 'shift', 'scale', 'random_state', 'Phase', 'Batch Size', 'Activation Func', 'Nodes', 'Layers', 'Optimizers', 'Weight Initializers', 'Dropout']

# Pulled the number of classes

# ----------------------------------------------------------------

# This allows us to generate all permutations of the parameters being tested in the model
import itertools

# --------------------------------------------------------------------------------------


# Let's get the packages we are going to need
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, LeakyReLU
from keras.callbacks import History
import tensorflow as tf
import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------

# RMSprop learning rate deafult, 'n_features', 'n_informative', 'n_classes', 'flip_y', 'class_sep', 'shift', 'scale', 'random_state', 'Phase', 'Batch Size', 'Activation Func', 'Nodes', 'Layers', 'Optimizers', 'Weight Initializers', 'Dropout']
settings1 = [.001, 17, 6, 8, .3, .4, 3, 2, 840780, 1]

final_param_list1 = []

# param_list_gen order is  units, activation function, batch size, layers
param_list_gen = [[8, 16, 32], ["sigmoid", "relu", "LeakyReLU"], [10, 20, 50], [1, 2]]
for element in itertools.product(*param_list_gen):
    final_param_list1.append(element)

# Create an empty list to hold the parameter output of each epoch
params_df = []

# --------  Model 1 - permutations of neurons, activation functions batch size and layers -------- #

for param in final_param_list1:

    q2model1 = Sequential()

    # hidden layer 1
    q2model1.add(Dense(param[2]))
    if param[1] != 'LeakyReLU':
        q2model1.add(Activation(param[1]))
    else:
        q2model1.add(LeakyReLU(alpha=0.1))

    if param[3] == 2:
        # hidden layer 2
        q2model1.add(Dense(param[2]))
        if param[1] != 'LeakyReLU':
            q2model1.add(Activation(param[1]))
        else:
            q2model1.add(LeakyReLU(alpha=0.1))

    # output layer
    q2model1.add(Dense(class_num, activation='softmax'))

    q2model1.compile(loss='sparse_categorical_crossentropy', optimizer='RMSProp', metrics=['accuracy'])

    history = q2model1.fit(X1, y1, epochs=epochs, batch_size=param[0])

    # Let's print out the variables to keep track
    print(param)


    params_df.append(settings1 + list(param) + ['RMSProp'] + ['NaN'] + ['NaN'] + history.history['acc'])



for i in range(1, epochs+1):
    epoch_num = 'E'+str(i)
    columns_labels.append(epoch_num)

df = pd.DataFrame(params_df, columns=columns_labels)
df.to_csv('phase1.csv', index=False)


########################################################
#                        PHASE 2                       #
########################################################

#
settings2 = [17, 6, 8, .3, .4, 3, 2, 840780, 2, 32, 'ReLU', 50, 1]

final_param_list2 = []


# param_list_gen order is  optimizers, weight initializers, dropout rate
param_list_gen2 = [['SGD', 'RMSProp', 'Adam'], ['glorot_uniform', 'glorot_uniform', 'he_uniform', 'he_normal'], [0.2, 0.4, 0.5]]
for element in itertools.product(*param_list_gen2):
    final_param_list2.append(element)

# Create an empty list to hold the parameter output of each epoch
params_df2 = []

# --------  Model 2 - permutations of optimizers, weight initializers and dropout rate -------- #

for param2 in final_param_list2:

    q2model2 = Sequential()

    # add dropout rate
    q2model2.add(Dropout(param2[2]))
    # hidden layer 1
    q2model2.add(Dense(50, kernel_initializer=param2[1]))

    # output layer
    q2model2.add(Dense(class_num, activation='softmax'))

    q2model2.compile(loss='sparse_categorical_crossentropy', optimizer=param2[0], metrics=['accuracy'])

    history = q2model2.fit(X1, y1, epochs=epochs, batch_size=32)

    # This is capturing the default learning rates for each optimizer.
    learning_rate = 0

    if param2[0] == 'SGD':
        learning_rate = 0.01
    else:
        learning_rate = 0.001

    # Let's print out the variables to keep track
    print(param2, learning_rate)

    # here we gather up all of the data that needs to be entered into the cvs file. The hard coded values represent the set
    params_df2.append([learning_rate] + settings2 + list(param2) + history.history['acc'])



columns_labels2 = ['Learning_Rate', 'n_features', 'n_informative', 'n_classes', 'flip_y', 'class_sep', 'shift', 'scale', 'random_state', 'Phase', 'Batch Size', 'Activation Func', 'Nodes', 'Layers', 'Optimizers', 'Weight Initializers', 'Dropout']

for i in range(1, epochs+1):
    epoch_num = 'E'+str(i)
    columns_labels2.append(epoch_num)

df2 = pd.DataFrame(params_df2, columns=columns_labels2)
df2.to_csv('phase2.csv', index=False)

dfs = [df, df2]

df_final = pd.concat(dfs)
df_final.to_csv( 'phase1-2.csv', index=False)