# Ideological Book Corpus

I got this data from https://github.com/klcoltin/Classification-of-News-Articles. "loadIBC.py" and "treeUtil.py" come from this same source (I modified these files for Python 3.x).

## Reference

Yanchuan Sim, Brice Acree, Justin Gross, and Noah Smith. Measuring Ideological Proportions in Political Speeches. Empirical Methods in Natural Language Processing, 2013.

https://www.cs.cmu.edu/~nasmith/papers/sim+acree+gross+smith.emnlp13.pdf

## To Run This Code:

mkdir output

python3 wse_loadIBC.py

python3 split_into_train_val_and_test.py

## Output

If you run this code, you will find in the output directory files containing X- and Y- values for the full set, a test set, a training set, and a validation set. Change the random seed in "split_into_train_val_and_test.py" to get different selections.


