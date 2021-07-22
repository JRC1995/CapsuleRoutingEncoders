import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
import torch as T
import random
import numpy as np


def loadAdamW(model, config):
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'embedding']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': config.wd},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]

    optimizer = T.optim.AdamW(optimizer_grouped_parameters, lr=config.lr)
    return optimizer


def display(texts, predictions, labels):

    N = len(texts)
    j = random.choice(np.arange(N).tolist())

    display_text = texts[j]
    display_prediction = predictions[j]
    display_gold = labels[j]

    print("\n\nExample Prediction\n")
    print("Text: {}\n".format(display_text))
    print("PREDICTION: {} | GOLD: {}\n".format(display_prediction, display_gold))


def cross_entropy(logits, labels, label_weights):

    ce_loss = nn.CrossEntropyLoss(reduction='none')

    loss = ce_loss(logits, labels)*label_weights

    return loss.mean()


def multi_binary_cross_entropy(model, logits, labels):

    labels = labels.float().view(-1)

    bce_loss = nn.BCELoss(reduction='none')
    bce = bce_loss(logits.view(-1), labels)
    #recall_weights = (labels*label_weights + (1-labels))
    # bce = bce*label_masks*recall_weights  # *((1-labels)*label_weights + labels)
    model.loss_all = bce

    return bce.mean()


def multi_binary_margin_loss(model, logits, labels):
    # print("hello")
    labels = labels.float().view(-1)
    logits = logits.view(-1)

    loss = labels * (T.max(T.tensor(0.0), 0.9 - logits)**2) + \
        0.25 * (1.0 - labels) * (T.max(T.tensor(0.0), logits - 0.1)**2)

    model.loss_all = loss

    return loss.mean()


def predict(model, text_idx, labels, input_mask, device, loss='BCE', train=True):

    # print(label_weights)

    with T.no_grad():

        text_idx = T.tensor(text_idx).long().to(device)
        labels = T.tensor(labels).long().to(device)
        input_mask = T.tensor(input_mask).float().to(device)

    if train:

        model = model.train()
        logits = model(text_idx, input_mask)

    else:
        model = model.eval()
        logits = model(text_idx, input_mask)

    predictions = logits.view(-1).detach().cpu().numpy()
    predictions = np.where(predictions > 0.5, 1, 0)

    # print(predictions.shape)
    # print(predictions)

    if loss == 'BCE':
        loss = multi_binary_cross_entropy(model, logits, labels)
    else:
        loss = multi_binary_margin_loss(model, logits, labels)

    #loss = T.tensor(0.0)

    T.cuda.empty_cache()

    return predictions, loss


def multi_micro_metrics(predictions, labels, verbose=False):

    tp = 0
    pred_len = 0
    gold_len = 0
    total_accuracy = 0

    correct = 0
    total = 0

    for prediction, label in zip(predictions, labels):
        if int(prediction) == int(label):
            correct += 1
            if int(prediction) == 1:
                tp += 1

        total += 1
        if int(label) == 1:
            gold_len += 1
        if int(prediction) == 1:
            pred_len += 1
    precision = tp/pred_len if pred_len > 0 else 0
    recall = tp/gold_len if gold_len > 0 else 0
    accuracy = correct/total if total > 0 else 0

    if verbose:
        print("\n")
        print("pred_len", pred_len)
        print("gold_len", gold_len)
        print("precision", precision)
        print("recall", recall)

    return precision, recall, accuracy


def multi_metrics(predictions, labels, idx2labels, verbose=False):

    total_precision = 0
    total_recall = 0
    total_accuracy = 0

    correct = 0
    total = 0

    for id in idx2labels:
        tp = 0

        pred_len = 0
        gold_len = 0

        for prediction, label in zip(predictions, labels):
            if label == id:
                if prediction == label:
                    correct += 1
                    tp += 1
                gold_len += 1
                total += 1
            if prediction == id:
                pred_len += 1
        precision = tp/pred_len if pred_len > 0 else 0
        recall = tp/gold_len if gold_len > 0 else 0
        accuracy = correct/total if total > 0 else 0

        if verbose:
            print("\n")
            print("Label", idx2labels[id])
            print("pred_len", pred_len)
            print("gold_len", gold_len)
            print("precision", precision)
            print("recall", recall)

        total_precision += precision
        total_recall += recall
        total_accuracy += accuracy

    precision = total_precision/len(idx2labels)
    recall = total_recall/len(idx2labels)
    accuracy = total_accuracy/len(idx2labels)

    if verbose:
        print("\n")
        print("avg precision", precision)
        print("avg recall", recall)
        print("\n")

    return precision, recall, accuracy


def compute_F1(precision, recall):
    F1 = (2*precision*recall)/(precision+recall) if (precision+recall) > 0 else 0
    return F1
