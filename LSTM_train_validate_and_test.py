#
# load useful libraries
#
import pprint as pp
import numpy as np
import os
import pickle
import sys
import argparse
import datetime

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

#
# load useful local libraries
#
import LSTM_utilities as ut

#
# get current date and time
#
timestamp = str(datetime.datetime.now()).replace(' ', '-').replace(':', '-')

#
# parse command line arguments
#
parser = argparse.ArgumentParser(description='Create a list of texts given a list of URLs.')
parser.add_argument('--output-directory', type=str, required=True, help='Name of directory to place output (will be erased/created).')
parser.add_argument('--input-file-directory', type=str, required=True, help='Name of directory containing x_train.txt, y_train.txt, x_val.txt, y_val.txt, x_test.txt, and y_test.txt.')
parser.add_argument('--GloVe-file', type=str, help='Filename of GloVe file to use.')
parser.add_argument('--max-sequence-length', type=int, help='Maximum sequence length.', required=True)
parser.add_argument('--number-of-layers-to-try', type=str, help='Comma-delimited list of number of layers to try, e.g. 1,2,3.', required=True)
parser.add_argument('--number-of-cells-to-try', type=str, help='Comma-delimited list of number of cells to try per layer, e.g. 10,15,30,70,128,200.', required=True)
parser.add_argument('--epochs', type=int, help='Number of epochs to use for gradient descent.', required=True)
parser.add_argument('--embeddings-to-try', type=str, help='One of [learned], [GloVe], or [learned,GloVe], e.g. --embeddings-to-try learned,GloVe', required=True)

args = parser.parse_args()

output_directory = args.output_directory + '-' + timestamp
input_directory = args.input_file_directory
GloVe_file = args.GloVe_file  #'/Users/emily/Desktop/data/NLP/GloVe/glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = args.max_sequence_length
layers = [int(x) for x in args.number_of_layers_to_try.split(',')]
number_of_cells_to_try = [int(x) for x in args.number_of_cells_to_try.split(',')]
epochs = args.epochs

embeddings_to_try = args.embeddings_to_try.split(',')

if 'GloVe' in embeddings_to_try and GloVe_file == None:
    print('You must specify --GloVe-file <filename> if you intend to use the GloVe embeddings. Exiting.')
    sys.exit(-1)

#
# useful hardcoded settings
#
x_train_filename = input_directory + '/x_train.txt'
y_train_filename = input_directory + '/y_train.txt'
x_val_filename = input_directory + '/x_val.txt'
y_val_filename = input_directory + '/y_val.txt'
x_test_filename = input_directory + '/x_test.txt'
y_test_filename = input_directory + '/y_test.txt'

output_scores_file = 'scores.pickled'

MODEL_CP_DIR = output_directory + '/checkpoints'
TENSORBOARD_BASE_DIR = output_directory + '/tensorboard'

#
# erase and create the output directory
#
if os.path.isdir(output_directory):
    os.system('rm -R ' + output_directory)
os.system('mkdir ' + output_directory)
os.system('mkdir ' + MODEL_CP_DIR)
os.system('mkdir ' + MODEL_CP_DIR + '/weights')
os.system('mkdir ' + MODEL_CP_DIR + '/images')
os.system('mkdir ' + TENSORBOARD_BASE_DIR)

#
# load files
#
x_train, y_train = ut.load_x_and_y_lists(x_train_filename, y_train_filename)
x_val, y_val = ut.load_x_and_y_lists(x_val_filename, y_val_filename)
x_test, y_test = ut.load_x_and_y_lists(x_test_filename, y_test_filename)

#
# make y lists NumPy arrays
#
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)
y_test = np.asarray(y_test)

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
x_test = tokenizer.texts_to_sequences(x_test)
x_test = pad_sequences(x_test, maxlen=MAX_SEQUENCE_LENGTH)

#
# create word index
#
word_index = tokenizer.word_index


#
# iterate through the embedding methods
#
score_results = {}
for embedding_method in embeddings_to_try:
    score_results[embedding_method] = {}

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
    # iterate through the layers
    #

    for number_of_layers in layers:

        score_results[embedding_method][number_of_layers] = {}

        for number_of_cells in number_of_cells_to_try:

            score_results[embedding_method][number_of_layers][number_of_cells] = None

            #
            # initialize model
            #
            model = ut.create_LSTM_model_framework(embedding_layer, number_of_layers, MAX_SEQUENCE_LENGTH, number_of_cells)

            #
            # train
            #
            results_file = ut.train_and_evaluate(x_train, y_train, x_val, y_val, model, number_of_layers, number_of_cells, embedding_method, MODEL_CP_DIR, TENSORBOARD_BASE_DIR, epochs=epochs)

            #
            # produce ROC
            #
            y_scores = model.predict(x_test)
            roc_filename = results_file.replace('/weights/', '/images/') + '.png'
            ut.compute_and_plot_ROC(y_test, y_scores, roc_filename)

            #
            # test
            #
            scores = model.evaluate(x_test, y_test, verbose=1)
            score_results[embedding_method][number_of_layers][number_of_cells] = {
                'scores' : scores,
                'results_file' : results_file,
                'ROC' : roc_filename,
                }

            print()
            pp.pprint(score_results)
            print()

            with open(output_directory + '/' + output_scores_file, 'wb') as f:
                pickle.dump(score_results, f)

