import csv
import sys
import json
import pickle
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
import numpy as np

nlp = English()
# Create a blank Tokenizer with just the English vocab
tokenizer = Tokenizer(nlp.vocab)


def _read_tsv(input_file, quotechar=None):

    with open(input_file, "r") as f:
        reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
        lines = []
        for line in reader:
            if sys.version_info[0] == 2:
                line = list(str(cell, 'utf-8') for cell in line)
            lines.append(line)
        return lines


train_lines = _read_tsv("../data/Reuters/train.tsv")
dev_lines = _read_tsv("../data/Reuters/dev.tsv")
test_lines = _read_tsv("../data/Reuters/test.tsv")


def process_class(c):
    return [float(x) for x in c]


train_classes = [process_class(line[0]) for line in train_lines]
dev_classes = [process_class(line[0]) for line in dev_lines]
test_classes = [process_class(line[0]) for line in test_lines]


def tokenize(text):
    return [token.text for token in tokenizer(text)]


train_texts = [line[1] for line in train_lines]
dev_texts = [line[1] for line in dev_lines]
test_texts = [line[1] for line in test_lines]

train_texts_vec = [tokenize(line[1]) for line in train_lines]
dev_texts_vec = [tokenize(line[1]) for line in dev_lines]
test_texts_vec = [tokenize(line[1]) for line in test_lines]

vocab2count = {}
for text in train_texts_vec:
    for word in text:
        vocab2count[word] = vocab2count.get(word, 0)+1

word_vec_dim = 300


def loadEmbeddings(filename):
    vocab2embd = {}

    global word_vec_dim

    with open(filename) as infile:
        i = 0
        for line in infile:
            if i != 0:
                row = line.strip().split(' ')
                word = row[0]
                embd = np.asarray(row[1:], np.float32)
                vocab2embd[word] = embd
            i = 1

    print('Embedding Loaded.')
    return vocab2embd


vocab2embd = loadEmbeddings("../embeddings/word2vec/word2vec.txt")

vocab2idx = {'<PAD>': 0, '<UNK>': 1}
embeddings = [np.zeros(word_vec_dim, np.float32), np.random.randn(word_vec_dim)]

c = 2
for word in vocab2count:
    if word in vocab2embd:
        vocab2idx[word] = c
        embeddings.append(vocab2embd[word])
        c += 1

embeddings = np.asarray(embeddings, np.float32)


def tovec(text):
    global vocab2idx
    return [vocab2idx.get(word, vocab2idx['<UNK>']) for word in text]


train_texts_vec = [tovec(line) for line in train_texts_vec]
dev_texts_vec = [tovec(line) for line in dev_texts_vec]
test_texts_vec = [tovec(line) for line in test_texts_vec]

# print(vocab2count)


with open("../processed_data/Reuters_vocab.pkl", "wb") as f:
    pickle.dump({'vocab2idx': vocab2idx, 'embeddings': embeddings}, f)

with open("../processed_data/Reuters_test.json", "w") as f:
    for text, text_idx, c in zip(test_texts, test_texts_vec, test_classes):
        f.write(json.dumps({"Text": text, "Text_idx": text_idx, "Class": c}))
        f.write("\n")

with open("../processed_data/Reuters_train.json", "w") as f:
    for text, text_idx, c in zip(train_texts, train_texts_vec, train_classes):
        f.write(json.dumps({"Text": text, "Text_idx": text_idx, "Class": c}))
        f.write("\n")

with open("../processed_data/Reuters_dev.json", "w") as f:
    for text, text_idx, c in zip(dev_texts, dev_texts_vec, dev_classes):
        f.write(json.dumps({"Text": text, "Text_idx": text_idx, "Class": c}))
        f.write("\n")
