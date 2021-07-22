import numpy as np
import math
import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F


device = T.device('cuda' if T.cuda.is_available() else 'cpu')

if device == T.device('cuda'):
    T.set_default_tensor_type(T.cuda.FloatTensor)
else:
    T.set_default_tensor_type(T.FloatTensor)


def cross_entropy(model, logits, labels, label_weights):

    # print(logits)
    # print(labels)

    ce_loss = nn.CrossEntropyLoss(reduction='none')

    loss = ce_loss(logits, labels)*label_weights

    return loss.mean()


def multi_binary_cross_entropy(model,
                               logits, labels):

    labels = labels.float().view(-1)

    bce_loss = nn.BCELoss(reduction='none')
    bce = bce_loss(logits.view(-1), labels)
    #recall_weights = (labels*label_weights + (1-labels))
    # bce = bce*label_masks*recall_weights  # *((1-labels)*label_weights + labels)

    return bce.mean()
