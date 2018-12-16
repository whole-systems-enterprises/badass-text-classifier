#
# load useful libraries
#
import pprint as pp
import numpy as np
import os
import pickle
import sys

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import LSTM_utilities as ut

#
# user settings
#
output_directory = 'output_predictions'

best_model = 'ideological-book-corpus/best_model_so_far/best_model_so_far.hdf5'
number_of_layers = 3
number_of_cells = 70
embedding_method = 'learned'

x_train_filename = 'ideological-book-corpus/output/x_train.txt'
y_train_filename = 'ideological-book-corpus/output/y_train.txt'
x_val_filename = 'ideological-book-corpus/output/x_val.txt'
y_val_filename = 'ideological-book-corpus/output/y_val.txt'

x_to_predict_filename = 'input.csv'

MAX_SEQUENCE_LENGTH = 1000


#
# load files
#
x_train, y_train = ut.load_x_and_y_lists(x_train_filename, y_train_filename)
x_val, y_val = ut.load_x_and_y_lists(x_val_filename, y_val_filename)
x_to_predict = ut.load_x_list(x_to_predict_filename)


#
# tokenize
#
tokenizer = Tokenizer()
texts_to_tokenize = x_train + x_val
tokenizer.fit_on_texts(texts_to_tokenize)
x_train = tokenizer.texts_to_sequences(x_train)
x_train = pad_sequences(x_train, maxlen=MAX_SEQUENCE_LENGTH)
x_val = tokenizer.texts_to_sequences(x_val)
x_val = pad_sequences(x_val, maxlen=MAX_SEQUENCE_LENGTH)

x_to_predict = tokenizer.texts_to_sequences(x_to_predict)
x_to_predict = pad_sequences(x_to_predict, maxlen=MAX_SEQUENCE_LENGTH)



#
# create word index
#
word_index = tokenizer.word_index



#
# create embeddings layer
#
if embedding_method == 'GloVe':
    embeddings_index = ut.index_glove_embeddings(GloVe_file)
    embedding_layer = ut.create_embedding_layer_GloVe(word_index, embeddings_index)
elif embedding_method == 'learned':
    embedding_layer = ut.create_embedding_layer_learned(word_index)
else:
    print('Unknown embedding method. Exiting.')
    sys.exit(0)

#
# create model
#
model = ut.create_LSTM_model_framework(embedding_layer, number_of_layers, MAX_SEQUENCE_LENGTH, number_of_cells)

#
# load weights
#
model.load_weights(best_model)

#
# make predictions 
#
y_predicted = [x[0] for x in model.predict(x_to_predict)]

#
# save predictions
#
with open(output_directory + '/y_predicted.pickled', 'wb') as f:
    pickle.dump(y_predicted, f)

