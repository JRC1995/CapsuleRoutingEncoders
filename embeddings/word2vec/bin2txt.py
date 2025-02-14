from argparse import ArgumentParser

from gensim.models.keyedvectors import KeyedVectors
import torch
from tqdm import tqdm


if __name__ == '__main__':
    #parser = ArgumentParser(description='Convert binary word2vec to txt')
    # parser.add_argument('input')
    # parser.add_argument('output')

    #args = parser.parse_args()
    model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin.gz', binary=True)
    model.save_word2vec_format('word2vec.txt', binary=False)
