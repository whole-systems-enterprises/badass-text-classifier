#
# load useful libraries
#
import pprint as pp
import numpy as np
import os
import pickle
import sys
import argparse

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

import LSTM_utilities as ut

#
# command line arguments
#
parser = argparse.ArgumentParser(description='Create a list of texts given a list of URLs.')
parser.add_argument('--x-to-predict-file', type=str, help='Path and name of file containing text to predict labels for, one text per line.', required=True)
parser.add_argument('--output-file', type=str, help='File to store predicted values for y, one per line.', required=True)
parser.add_argument('--number-of-layers', type=int, help='Number of layers in trained model.', required=True)
parser.add_argument('--number-of-cells', type=int, help='Number of cells per layer in the trained model.', required=True)
parser.add_argument('--model-file', type=str, help='Path and name of file containing trained model.', required=True)
parser.add_argument('--embedding-method', type=str, help='One of "GloVe" or "learned".', required=True)
parser.add_argument('--GloVe-file', type=str, help='Filename of GloVe file to use.')
parser.add_argument('--max-sequence-length', type=int, help='Maximum sequence length.', required=True)
parser.add_argument('--input-file-directory', type=str, required=True, help='Name of directory containing x_train.txt, y_train.txt, x_val.txt, y_val.txt, x_test.txt, and y_test.txt.')
args = parser.parse_args()

x_to_predict_filename = args.x_to_predict_file
best_model = args.model_file
number_of_layers = args.number_of_layers
number_of_cells = args.number_of_cells
embedding_method = args.embedding_method
GloVe_file = args.GloVe_file
MAX_SEQUENCE_LENGTH = args.max_sequence_length
input_directory = args.input_file_directory
output_filename = args.output_file

if embedding_method == 'GloVe' and GloVe_file == None:
    print('You must specify the GloVe embeddings file location with --GloVe-file <filename> if you want to use the GloVe embedding method. Exiting.')
    sys.exit(0)

#
# useful hardcoded settings
#
x_train_filename = input_directory + '/x_train.txt'
y_train_filename = input_directory + '/y_train.txt'
x_val_filename = input_directory + '/x_val.txt'
y_val_filename = input_directory + '/y_val.txt'


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
f = open(output_filename, 'w')
for yp in y_predicted:
    f.write(str(yp + '\n'))
f.close()


