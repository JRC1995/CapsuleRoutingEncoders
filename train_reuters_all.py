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


device = T.device('cuda' if T.cuda.is_available() else 'cpu')
if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


dataset = "Reuters"


embedding_filename = 'processed_data/Reuters_vocab.pkl'
training_filename = 'processed_data/Reuters_train.json'
validation_filename = 'processed_data/Reuters_dev.json'
classes_num = 90

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

val_classes = []
val_texts = []
val_texts_idx = []
with open(validation_filename, 'r') as file:
    for obj in file:
        row = json.loads(obj)
        val_classes.append(row['Class'])
        val_texts.append(row['Text'])
        val_texts_idx.append(row['Text_idx'])


def train(time, model_name):

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

    print("\n\nTraining Model: {}\n\n".format(model_name))

    Encoder = model_dict[model_name]

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

    load = 'n'  # input("\nLoad checkpoint? y/n: ")
    print("")
    checkpoint_path = "saved_params/{}_{}_{}.pt".format(dataset, model_name, time)

    if load.lower() == 'y':
        print('Loading pre-trained weights for the model...')
        checkpoint = T.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        past_epoch = checkpoint['past epoch']
        best_val_F1 = checkpoint['best F1']
        best_val_cost = checkpoint['best loss']
        impatience = checkpoint['impatience']
        print('\nRESTORATION COMPLETE\n')

    else:
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
                                        loss='margin',
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

            if n % display_step == 0:
                print("Validating Batch {}".format(n+1))

            with T.no_grad():

                predictions, loss = predict(model=model,
                                            text_idx=batch_texts_idx,
                                            labels=batch_labels,
                                            input_mask=batch_mask,
                                            device=device,
                                            loss='margin',
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

        print("\n\nVALIDATION\n\n")

        print("Epoch "+str(epoch)+":, Cost = " +
              "{:.3f}".format(avg_val_cost)+", F1 = " +
              "{:.3f}".format(val_F1)+", Accuracy = " +
              "{:.3f}".format(acc))

        flag = 0
        impatience += 1

        if avg_val_cost < best_val_cost:
            impatience = 0
            best_val_cost = avg_val_cost

        if val_F1 >= best_val_F1:
            impatience = 0
            best_val_F1 = val_F1
            flag = 1

        if flag == 1:

            T.save({
                'past epoch': epoch+1,
                'best loss': best_val_cost,
                'best F1': best_val_F1,
                'impatience': impatience,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, checkpoint_path)

            print("Checkpoint created!")

        print("\n")

        if impatience > patience:
            break


#"CNN", "CNN_att",
model_names = ["CNN", "CNN_att", "CNN_capsule",
               "CNN_heinsen_capsule", "CNN_DSA", "CNN_DSA_global",
               "CNN_PCaps",
               "CNN_custom", "CNN_custom_alpha_ablation", "CNN_custom_global",
               "CNN_custom2"]

for model_name in model_names:

    T.manual_seed(101)
    random.seed(101)
    T.backends.cudnn.deterministic = True
    T.backends.cudnn.benchmark = False
    np.random.seed(101)

    for time in range(5):
        print("\n\nCurrent Run: {}\n\n".format(time))
        train(time, model_name)
