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

#
# load useful local libraries
#
import LSTM_utilities as ut

#
# user settings
#
output_directory = 'LSTM_output_IBC'
x_train_filename = 'ideological-book-corpus/output/x_train.txt'
y_train_filename = 'ideological-book-corpus/output/y_train.txt'
x_val_filename = 'ideological-book-corpus/output/x_val.txt'
y_val_filename = 'ideological-book-corpus/output/y_val.txt'
x_test_filename = 'ideological-book-corpus/output/x_test.txt'
y_test_filename = 'ideological-book-corpus/output/y_test.txt'

output_scores_file = 'scores.pickled'

GloVe_file = '/Users/emily/Desktop/data/NLP/GloVe/glove.6B.100d.txt'
MAX_SEQUENCE_LENGTH = 1000

layers = [1, 2, 3]
number_of_cells_to_try = [10, 15, 30, 70, 128, 200]
embeddings_to_try = ['learned']

MODEL_CP_DIR = output_directory + '/checkpoints'
TENSORBOARD_BASE_DIR = output_directory + '/tensorboard'

epochs = 10

#
# crudely clear the way
#
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
for embedding_method in embeddings_to_try:

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
    score_results = {}
    for number_of_layers in layers:

        score_results[number_of_layers] = {}

        for number_of_cells in number_of_cells_to_try:

            score_results[number_of_layers][number_of_cells] = None

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
            score_results[number_of_layers][number_of_cells] = {
                'scores' : scores,
                'results_file' : results_file,
                'ROC' : roc_filename,
                }

            print()
            pp.pprint(score_results)
            print()

            with open(output_directory + '/' + output_scores_file, 'wb') as f:
                pickle.dump(score_results, f)

