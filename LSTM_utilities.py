#
# load useful libraries
#
import sys
import os
import numpy as np
from keras.layers import Embedding
from keras.layers import Dense, Input, LSTM, Activation, Dropout
from keras.models import Model
from keras.callbacks import TensorBoard, ModelCheckpoint
from time import localtime, strftime
from sklearn.metrics import roc_curve, auc

#
# compute ROC curve
#
# https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc.html
#
def compute_and_plot_ROC(y_known, y_score, filename):

    import matplotlib.pyplot as plt

    y_known = [int(x) for x in y_known]

    # crude
    y_score_list = []
    for s in y_score:
        y_score_list.append(s[0])

    fpr, tpr, thresholds = roc_curve(y_known, y_score_list)
    roc_auc = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC Curve (Area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.savefig(filename)
    plt.close()

#
# Load lists of X and Y values
#
# Each line is an entry, the count of lines in the two files must be the same, and there is no header
#
def load_x_and_y_lists(x_list_filename, y_list_filename):
    x_list = []
    f = open(x_list_filename)
    for line in f:
        line = line.strip()
        x_list.append(line)
    f.close()

    y_list = []
    f = open(y_list_filename)
    for line in f:
        line = line.strip()
        y_list.append(line)
    f.close()

    if len(x_list) != len(y_list):
        print('Lists are not the same length. Exiting')
        sys.exit(0)
    if len(x_list) == 0:
        print('Lists are empty. Exiting.')
        sys.exit(0)

    return x_list, y_list


def load_x_list(x_list_filename):
    x_list = []
    f = open(x_list_filename)
    for line in f:
        line = line.strip()
        x_list.append(line)
    f.close()
    return x_list


#
# index GloVe embeddings
#
# This function is (very) slightly modified from that found in https://github.com/natel9178/CS230-news-bias/blob/master/train.py
#
def index_glove_embeddings(filename):
    embeddings_index = {}
    with open(filename) as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

#
# Create embedding layer
#
# This function is (very) slightly modified from that found in https://github.com/natel9178/CS230-news-bias/blob/master/train.py
#
def create_embedding_layer_GloVe(word_index, embeddings_index, EMBEDDING_DIM=100):
    num_words = len(word_index) + 1

    embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    return Embedding(
        num_words,
        EMBEDDING_DIM,
        weights=[embedding_matrix],
        trainable=False
        )


#
# Create embedding layer
#
# This function is (very) slightly modified from that found in https://github.com/natel9178/CS230-news-bias/blob/master/train.py
#
def create_embedding_layer_learned(word_index, EMBEDDING_DIM=100):
    num_words = len(word_index) + 1
    return Embedding(
        num_words,
        EMBEDDING_DIM,
        )


#
# create LSTM model framework
#
# This function is heavily modified from that found in https://github.com/natel9178/CS230-news-bias/blob/master/train.py
#
def create_LSTM_model_framework(embedding_layer, number_of_layers, MAX_SEQUENCE_LENGTH, number_of_cells):
    sequence_input = Input(shape=(MAX_SEQUENCE_LENGTH,), dtype='int32')
    embedded_sequences = embedding_layer(sequence_input)

    X = embedded_sequences
    for i in range(number_of_layers-1):
        X = LSTM(number_of_cells, return_sequences=True)(X)
        X = Dropout(0.2)(X)
    X = LSTM(number_of_cells, return_sequences=False)(X)
    X = Dropout(0.2)(X)
    X = Dense(1)(X)
    preds = Activation('sigmoid')(X)

    return Model(sequence_input, preds)

#
# Train
#
# This function is heavily modified from that found in https://github.com/natel9178/CS230-news-bias/blob/master/train.py
#
def train_and_evaluate(x_train, y_train, x_val, y_val, model, number_of_layers, number_of_cells, embedding_method, MODEL_CP_DIR, TENSORBOARD_BASE_DIR, epochs=10):
    model_type = 'lstm'

    MODEL_CP_DIR = '{}layers-{}-cells-{}-embedding-method-{}-model-type-{}{}'.format(MODEL_CP_DIR + '/weights/', number_of_layers, number_of_cells, embedding_method, model_type, '-weights.best.hdf5')

    model.compile(
        loss='binary_crossentropy',
        optimizer='adam',
        metrics=['acc']
        )

    tensorboard = TensorBoard(log_dir=os.path.join(TENSORBOARD_BASE_DIR, "{}{}{}{}{}".format(number_of_layers, number_of_cells, embedding_method, model_type, strftime("%Y-%m-%d_%H-%M-%S", localtime()))))

    checkpoint = ModelCheckpoint(MODEL_CP_DIR, monitor='val_acc', verbose=1, save_best_only=True, mode='max')

    model.fit(
        x_train,
        y_train,
        batch_size=128,
        epochs=epochs,
        validation_data=(x_val, y_val), verbose=1, callbacks=[tensorboard, checkpoint]
        )

    return MODEL_CP_DIR
