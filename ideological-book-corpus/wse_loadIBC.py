import _pickle as cPickle

#
# user settings
#
output_directory = 'output'

if __name__ == '__main__':
    [lib, con, neutral] = cPickle.load(open('ibcData.pkl', 'rb'))

    # how to access sentence text
    print('Liberal examples (out of ', len(lib), ' sentences): ')
    for tree in lib[0:5]:
        print(tree.get_words())

    print('\nConservative examples (out of ', len(con), ' sentences): ')
    for tree in con[0:5]:
        print(tree.get_words())

    print('\nNeutral examples (out of ', len(neutral), ' sentences): ')
    for tree in neutral[0:5]:
        print(tree.get_words())

    # how to access phrase labels for a particular tree
    ex_tree = lib[0]

    print('\nPhrase labels for one tree: ')

    # see treeUtil.py for the tree class definition
    for node in ex_tree:

        # remember, only certain nodes have labels (see paper for details)
        if hasattr(node, 'label'):
            print(node.label, ': ', node.get_words())


    #
    # write contents to files
    #
    f_x = open(output_directory + '/x_full.txt', 'w')
    f_y = open(output_directory + '/y_full.txt', 'w')
    for tree in lib:
        f_x.write(tree.get_words() + '\n')
        f_y.write('0' + '\n')
    for tree in con:
        f_x.write(tree.get_words() + '\n')
        f_y.write('1' + '\n')
    f_x.close()
    f_y.close()
