#
# load useful libraries
#
import numpy as np
import random
import pprint as pp

#
# user settings
#
random.seed(400523)

output_directory = 'output'

train_val_test_proportion = {
    'train' : 0.8,
    'val' : 0.1,
    'test' : 0.1,
}

#
# load x_ and y_full
#
f = open(output_directory + '/x_full.txt')
x_full = []
for line in f:
    line = line.strip()
    x_full.append(line)
f.close()
f = open(output_directory + '/y_full.txt')
y_full = []
for line in f:
    line = line.strip()
    y_full.append(line)
f.close()

#
# shuffle
#
combined = list(zip(x_full, y_full))
random.shuffle(combined)
x_full[:], y_full[:] = zip(*combined)

#
# split into train, val, and test sets
#
key_list = sorted(list(train_val_test_proportion.keys()))
position = 0
split_x = {}
split_y = {}
for key in key_list:
    old_position = position
    position += int(round(train_val_test_proportion[key] * float(len(x_full))))
    if len(x_full) - position <= 3:
        position = len(x_full)
    split_x[key] = x_full[old_position:position]
    split_y[key] = y_full[old_position:position]

#
# make x/y_train/val/dev 
#
for key in key_list:
    f_x = open(output_directory + '/x_' + key + '.txt', 'w')
    f_y = open(output_directory + '/y_' + key + '.txt', 'w')
    for x, y in zip(split_x[key], split_y[key]):
        f_x.write(x + '\n')
        f_y.write(y + '\n')
    f_x.close()
    f_y.close()





