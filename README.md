# badass-text-classifier

## A badass text classifier based on machine learning

Want to classify celebrity tweets? Scientific journal abstracts? Movie reviews? Then use this binary classifier to train a deep learning model from your labeled texts to predict labels for unknown text.

## Intense gratitude (credit where it is due)

This work drew heavily on the student work provided at https://github.com/natel9178/CS230-news-bias.

See the "ideological-book-corpus" directory for citations for that dataset.

## Requirements

To train a model, and predict from it, you need:

- Python 3
- NumPy
- Keras
- TensorFlow
- MatplotLib

To use the optional GloVe embeddings [1], download the pre-trained vectors from https://nlp.stanford.edu/projects/glove/.

To use the optional "prepare_x_list_from_URL_list" utility, you also need:

- requests
- boilerpipe

## Data Preparation

In a directory (suppose for the sake of example called "input-files"), you need six files:

- x_train.txt
- y_train.txt
- x_val.txt
- y_val.txt
- x_test.txt
- y_test.txt

where "train", "val", and "tests" correspond to your training, validation, and testing sets, respectively. The "x_" files contain text cases, one per line, while the "y_" files contain (corresponding by line number) either a "0" or "1" indicating the known class.

Deciding how to split your data set is somewhat of an art; I typically use 80% for training, and 10% each for validation and testing. However, this is just my habit, not a rigorous recommendation.

## Training

Suppose you have the above prepared files contained in a subdirectory called "input", and that you want to write all output to a subdirectory called "output-<timestamp>". Furthermore, suppose you want to use a maximum sequence length of 1000, to try between one and three layers, and [10, 15, 30, 70, 128, and 200] cells per layer. Finally, suppose you want to use both the GloVe word embeddings index and a word embeddings index learned from the data, and to run the gradient descent over 100 epochs. The following command will deliver these results:

```
python3 LSTM_train_validate_and_test.py --output-directory output --input-file-directory input --max-sequence-length 1000 --number-of-layers-to-try 1,2,3 --number-of-cells-to-try 10,15,30,70,128,200 --epochs 100 --embeddings-to-try learned,GloVe --GloVe-file /Users/emily/Desktop/data/NLP/GloVe/glove.6B.100d.txt
```

The resulting models will appear in the "output-<timestamp>/checkpoints/weights" directory, and ROC plots resulting from running the models on the test set will appear in the "output-<timestamp>/checkpoints/images" directory.

## Running the utility to retrieve webpages in text form given a list of URLs

In the "prepare_x_list_from_URL_list_directory", suppose you have a list of URLs stored in a file "list_of_URLs.txt", one per line. Then the command:

```
python3 prepare_x_list_from_URL_list.py --url-list-file list_of_URLs.txt --output-directory output --timeout 10
```

retrieves the text files and places the results in "output". 

## References

1. Jeffrey Pennington, Richard Socher, and Christopher D. Manning. 2014. GloVe: Global Vectors for Word Representation. https://nlp.stanford.edu/pubs/glove.pdf




