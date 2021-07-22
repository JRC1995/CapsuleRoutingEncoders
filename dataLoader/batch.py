import copy
import numpy as np
import random
import re
import pickle
import os
import logging
logging.basicConfig(level=logging.CRITICAL)


def batcher(texts, texts_idx, labels, PAD, batch_size, buckets=10, seed=None):

    if seed is not None:
        random.seed(seed)

    true_seq_lens = [len(text_idx) for text_idx in texts_idx]

    def reorder(items, idx):
        return [items[i] for i in idx]

    # sorted in descending order after flip
    sorted_idx = np.flip(np.argsort(true_seq_lens), 0).tolist()
    sorted_texts = reorder(texts, sorted_idx)
    sorted_texts_idx = reorder(texts_idx, sorted_idx)
    sorted_labels = reorder(labels, sorted_idx)

    data_len = len(sorted_texts)

    #print("Sample size: ", data_len)

    bucket_size = data_len//buckets

    buckets = []
    c = 0
    while c < data_len:
        start = c
        end = c+bucket_size
        if end > data_len:
            end = data_len
        bucket = [sorted_texts[start:end],
                  sorted_texts_idx[start:end],
                  sorted_labels[start:end]]
        buckets.append(bucket)
        c = end

    random.shuffle(buckets)

    for bucket in buckets:
        b_texts, b_texts_idx, b_labels = bucket

        bucket_len = len(b_texts)
        idx = [i for i in range(bucket_len)]
        random.shuffle(idx)

        b_texts = reorder(b_texts, idx)
        b_texts_idx = reorder(b_texts_idx, idx)
        b_labels = reorder(b_labels, idx)

        i = 0

        while i < bucket_len:

            incr = batch_size
            if i+batch_size > bucket_len:
                incr = bucket_len-i

            batch_texts = []
            batch_texts_idx = []
            batch_masks = []
            batch_labels = []

            max_len = max([len(b_texts_idx[i]) for i in range(i, i+incr)])

            for j in range(i, i + incr):

                text = b_texts[j]
                text_idx = b_texts_idx[j]
                label = b_labels[j]

                text_len = len(text_idx)
                attention_mask = [1]*text_len

                while len(text_idx) < max_len:
                    text_idx.append(PAD)
                    attention_mask.append(0)

                batch_texts.append(text)
                batch_texts_idx.append(text_idx)
                batch_labels.append(label)
                batch_masks.append(attention_mask)

            i += incr

            batch_texts_idx = np.asarray(batch_texts_idx, np.int)
            batch_masks = np.asarray(batch_masks, np.int)
            batch_labels = np.asarray(batch_labels, np.int)

            yield batch_texts, batch_texts_idx, batch_labels, batch_masks
