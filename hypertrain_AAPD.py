
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
from functools import partial
import json
from hyperopt import fmin, tpe, hp, STATUS_OK

device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)

search_space = {}


search_space["CNN_custom"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                              "dropout": hp.uniform("dropout", 0.0, 0.5),
                              "output_channel": hp.choice("output_channel", [64, 128]),
                              "C1": hp.choice("C1", [1, 3]),
                              "D": hp.choice("D", [16, 32, 64]),
                              "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN_custom2"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                               "dropout": hp.uniform("dropout", 0.0, 0.5),
                               "output_channel": hp.choice("output_channel", [64, 128]),
                               "C1": hp.choice("C1", [1, 3]),
                               "D": hp.choice("D", [16, 32, 64]),
                               "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN_DSA"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                           "dropout": hp.uniform("dropout", 0.0, 0.5),
                           "output_channel": hp.choice("output_channel", [64, 128]),
                           "C1": hp.choice("C1", [1, 3]),
                           "D": hp.choice("D", [16, 32, 64]),
                           "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN_capsule"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                               "dropout": hp.uniform("dropout", 0.0, 0.5),
                               "output_channel": hp.choice("output_channel", [64, 128]),
                               "C1": hp.choice("C1", [1, 3]),
                               "D": hp.choice("D", [16, 32, 64]),
                               "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN_heinsen_capsule"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                                       "dropout": hp.uniform("dropout", 0.0, 0.5),
                                       "output_channel": hp.choice("output_channel", [64, 128]),
                                       "C1": hp.choice("C1", [1, 3]),
                                       "D": hp.choice("D", [16, 32, 64]),
                                       "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN_PCaps"] = {"lr": hp.uniform("lr", 1e-4, 1e-2),
                             "dropout": hp.uniform("dropout", 0.0, 0.5),
                             "output_channel": hp.choice("output_channel", [64, 128]),
                             "C1": hp.choice("C1", [1, 3]),
                             "D": hp.choice("D", [16, 32, 64]),
                             "wd": hp.uniform("wd", 1e-4, 1e-2)}

search_space["CNN"] = {"lr": hp.uniform("lr", 1e-3, 1e-1),
                       "dropout": hp.uniform("dropout", 0.1, 0.3),
                       "output_channel": hp.choice("output_channel", [64, 128]),
                       "wd": hp.uniform("wd", 1e-5, 1e-2)}

search_space["CNN_att"] = {"lr": hp.uniform("lr", 1e-3, 1e-1),
                           "dropout": hp.uniform("dropout", 0.1, 0.3),
                           "att_dim": hp.choice("att_dim", [64, 128]),
                           "output_channel": hp.choice("output_channel", [64, 128]),
                           "wd": hp.uniform("wd", 1e-5, 1e-2)}

config_dict = {'CNN': CNN_config,
               'CNN_att': CNN_att_config,
               'CNN_capsule': CNN_capsule_config,
               'CNN_heinsen_capsule': CNN_heinsen_capsule_config,
               'CNN_DSA': CNN_DSA_config,
               'CNN_PCaps': CNN_PCaps_config,
               'CNN_custom': CNN_custom_config,
               'CNN_custom2': CNN_custom2_config}

model_dict = {'CNN': CNN,
              'CNN_att': CNN_att,
              'CNN_capsule': CNN_capsule,
              'CNN_heinsen_capsule': CNN_heinsen_capsule,
              'CNN_DSA': CNN_DSA,
              'CNN_PCaps': CNN_PCaps,
              'CNN_custom': CNN_custom,
              'CNN_custom2': CNN_custom2}


embedding_filename = 'processed_data/AAPD_vocab.pkl'
training_filename = 'processed_data/AAPD_train.json'
validation_filename = 'processed_data/AAPD_dev.json'
classes_num = 54

with open(embedding_filename, 'rb') as file:
    data = pickle.load(file)
vocab2idx = data['vocab2idx']
embeddings = data['embeddings']
PAD = vocab2idx['<PAD>']

train_classes = []
train_texts = []
train_texts_idx = []
with open(training_filename, 'r') as file:
    for obj in file:
        row = json.loads(obj)
        train_classes.append(row['Class'])
        train_texts.append(row['Text'])
        train_texts_idx.append(row['Text_idx'])

train_idx = [i for i in range(len(train_texts))]
train_idx = random.sample(train_idx, k=20000)

train_texts = [train_texts[i] for i in train_idx]
train_texts_idx = [train_texts_idx[i] for i in train_idx]
train_classes = [train_classes[i] for i in train_idx]

val_classes = []
val_texts = []
val_texts_idx = []
with open(validation_filename, 'r') as file:
    for obj in file:
        row = json.loads(obj)
        val_classes.append(row['Class'])
        val_texts.append(row['Text'])
        val_texts_idx.append(row['Text_idx'])


def train(args, model_name):

    global config_dict
    global model_dict
    global train_classes
    global train_texts
    global train_texts_idx
    global val_classes
    global val_texts
    global val_texts_idx
    global PAD
    global embeddings
    global vocab2idx
    global classes_num
    global device

    T.manual_seed(101)
    random.seed(101)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(101)

    config = config_dict[model_name]
    config = config()

    print("HYPERPARAMTERS:")
    print(args)
    print("\n\n")

    config.lr = args["lr"]
    config.wd = args["wd"]
    config.output_channel = args["output_channel"]
    config.dropout = args["dropout"]
    config.epochs = 5

    if 'capsule' in model_name:
        config.D = args["D"]
        config.C1 = args["C1"]
    elif 'att' in model_name:
        config.att_dim = args["att_dim"]

    accu_step = config.total_batch_size//config.train_batch_size

    print("\n\nTraining Model: {}\n\n".format(model_name))

    Encoder = model_dict[model_name]

    model = Encoder(embeddings=embeddings,
                    pad_idx=PAD,
                    classes_num=54,
                    config=config,
                    device=device)

    model = model.to(device)

    optimizer = loadAdamW(model, config)

    display_step = 100  # 100
    example_display_step = 500
    patience = 5

    past_epoch = 0
    best_val_cost = math.inf
    best_val_F1 = -math.inf
    impatience = 0

    for epoch in range(past_epoch, config.epochs):

        total_train_loss = 0
        total_F1 = 0
        i = 0

        for batch_texts, batch_texts_idx, \
            batch_labels, batch_mask in batcher(train_texts,
                                                train_texts_idx,
                                                train_classes,
                                                PAD,
                                                config.train_batch_size):

            predictions, loss = predict(model=model,
                                        text_idx=batch_texts_idx,
                                        labels=batch_labels,
                                        input_mask=batch_mask,
                                        device=device,
                                        train=True)

            loss = loss/accu_step

            loss.backward()

            if (i+1) % accu_step == 0:
                # Update accumulated gradients
                optimizer.step()
                optimizer.zero_grad()

            labels = batch_labels.reshape((-1)).tolist()

            prec, rec, acc = multi_micro_metrics(predictions.tolist(), labels)
            F1 = compute_F1(prec, rec)

            cost = loss.item()

            if i % display_step == 0:

                print("Iter "+str(i)+", Cost = " +
                      "{:.3f}".format(cost)+", F1 = " +
                      "{:.3f}".format(F1)+", Accuracy = " +
                      "{:.3f}".format(acc))

            i += 1
            # T.cuda.empty_cache()

        print("\n\n")

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
            """
            if n % display_step == 0:
                print("Validating Batch {}".format(n+1))
            """

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

        """

        print("\n\nVALIDATION\n\n")

        print("Epoch "+str(epoch)+":, Cost = " +
              "{:.3f}".format(avg_val_cost)+", F1 = " +
              "{:.3f}".format(val_F1)+", Accuracy = " +
              "{:.3f}".format(acc))
        """

        impatience += 1

        if avg_val_cost < best_val_cost:
            impatience = 0
            best_val_cost = avg_val_cost

        if val_F1 >= best_val_F1:
            impatience = 0
            best_val_F1 = val_F1

    return {'loss': -best_val_F1, 'status': STATUS_OK}


model_names = ["CNN", "CNN_att", "CNN_capsule",
               "CNN_heinsen_capsule", "CNN_DSA",
               "CNN_PCaps", "CNN_custom", "CNN_custom2"]


for model_name in model_names:

    fmin_train = partial(train, model_name=model_name)
    best = fmin(fmin_train,
                space=search_space[model_name],
                algo=tpe.suggest,
                max_evals=50)

    with open("configs/hyperopt_search/{}_AAPD_config.json".format(model_name), "w") as fp:
        d = {}
        for key in best:
            d[key] = str(best[key])
        json.dump(d, fp)
