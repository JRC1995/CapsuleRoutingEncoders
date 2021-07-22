import sys
from configs.reuters_args import *
import json
from models import *
from dataLoader.batch import batcher
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T
from utils import *
import argparse
import math
import random
import numpy as np
import pickle
import logging
logging.disable(logging.CRITICAL)


parser = argparse.ArgumentParser(description='Model Name and stuff')
parser.add_argument('--model', type=str, default="CNN_custom2")
flags = parser.parse_args()

device = T.device('cuda' if T.cuda.is_available() else 'cpu')
if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


model_name = flags.model
dataset = "Reuters"

config_dict = {'CNN': CNN_config,
               'CNN_att': CNN_att_config,
               'CNN_capsule': CNN_capsule_config,
               'CNN_heinsen_capsule': CNN_heinsen_capsule_config,
               'CNN_DSA': CNN_DSA_config,
               'CNN_DSA_global': CNN_DSA_config,
               'CNN_PCaps': CNN_PCaps_config,
               'CNN_custom': CNN_custom_config,
               'CNN_custom_alpha_ablation': CNN_custom_config,
               'CNN_custom_global': CNN_custom_config,
               'CNN_custom2': CNN_custom2_config}

model_dict = {'CNN': CNN,
              'CNN_att': CNN_att,
              'CNN_capsule': CNN_capsule,
              'CNN_heinsen_capsule': CNN_heinsen_capsule,
              'CNN_DSA': CNN_DSA,
              'CNN_DSA_global': CNN_DSA_global,
              'CNN_PCaps': CNN_PCaps,
              'CNN_custom': CNN_custom,
              'CNN_custom_alpha_ablation': CNN_custom_alpha_ablation,
              'CNN_custom_global': CNN_custom_global,
              'CNN_custom2': CNN_custom2}

config = config_dict[model_name]
config = config()
accu_step = config.total_batch_size//config.train_batch_size

print("\n\nTesting Model: {}\n\n".format(model_name))


Encoder = model_dict[model_name]

embedding_filename = 'processed_data/Reuters_vocab.pkl'
test_filename = 'processed_data/Reuters_test.json'
classes_num = 90

with open(embedding_filename, 'rb') as file:
    data = pickle.load(file)
vocab2idx = data['vocab2idx']
embeddings = data['embeddings']
PAD = vocab2idx['<PAD>']


val_classes = []
val_texts = []
val_texts_idx = []
with open(test_filename, 'r') as file:
    for obj in file:
        row = json.loads(obj)
        val_classes.append(row['Class'])
        val_texts.append(row['Text'])
        val_texts_idx.append(row['Text_idx'])


def test(time):

    model = Encoder(embeddings=embeddings,
                    pad_idx=PAD,
                    classes_num=classes_num,
                    config=config,
                    device=device)

    model = model.to(device)
    # model = T.nn.DataParallel(model)

    # Prepare optimizer
    optimizer = loadAdamW(model, config)

    display_step = 100  # 100
    example_display_step = 500
    patience = 5

    checkpoint_path = "saved_params/{}_{}_{}.pt".format(dataset, model_name, time)

    print('Loading pre-trained weights for the model...')
    checkpoint = T.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    total_val_cost = 0
    all_labels = []
    all_predictions = []
    n = 0

    for batch_texts, batch_texts_idx, \
        batch_labels, batch_mask in batcher(val_texts,
                                            val_texts_idx,
                                            val_classes,
                                            PAD,
                                            config.val_batch_size):

        if n % display_step == 0:
            print("Testing Batch {}".format(n+1))

        with T.no_grad():

            predictions, loss = predict(model=model,
                                        text_idx=batch_texts_idx,
                                        labels=batch_labels,
                                        input_mask=batch_mask,
                                        device=device,
                                        train=False)

            cost = loss.item()

            total_val_cost += cost
            labels = batch_labels.reshape(-1).tolist()
            all_labels += labels
            all_predictions += predictions.tolist()

        n += 1
        # T.cuda.empty_cache()

    prec, rec, acc = multi_micro_metrics(all_predictions,
                                         all_labels,
                                         verbose=False)
    val_F1 = compute_F1(prec, rec)
    avg_val_cost = total_val_cost/n

    print("\n\nTEST\n\n")

    print("Cost = " +
          "{:.3f}".format(avg_val_cost)+", F1 = " +
          "{:.3f}".format(val_F1)+", Accuracy = " +
          "{:.3f}".format(acc))

    return val_F1, acc


F1s = []
accs = []

for time in range(5):
    print("\n\nCurrent Run: {}\n\n".format(time))
    F1, acc = test(time)
    F1s.append(F1)
    accs.append(acc)

print("\n\n")

print("Mean-std F1: {} +- {}".format(np.mean(F1s), np.std(F1s)))
print("Mean-std Accuracy: {} +- {}".format(np.mean(accs), np.std(accs)))
print("Max F1: {}".format(np.max(F1s)))
print("Corresponding Accuracy: {}".format(accs[np.argmax(F1s)]))
