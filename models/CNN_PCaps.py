# Adapted from: https://github.com/castorini/hedwig/tree/master/models/kim_cnn

import torch as T
import torch.nn as nn
import torch.nn.functional as F
from models.modules.no_routing import routing


class Classifier(nn.Module):

    def __init__(self, embeddings, pad_idx, classes_num,
                 config, device):
        super(Classifier, self).__init__()

        trainable_embeddings = config.trainable_embeddings

        if trainable_embeddings:
            self.embeddings = nn.Parameter(T.tensor(embeddings).float().to(device))
        else:
            self.embeddings = T.tensor(embeddings).float().to(device)

        self.embedding_dropout = nn.Dropout(config.embedding_dropout)
        self.embedding_ones = T.ones(self.embeddings.size(0), 1).float().to(device)

        self.output_channel = config.output_channel
        words_dim = self.embeddings.size(-1)
        self.ks = 3  # There are three conv nets here
        self.pad_idx = pad_idx
        self.classes_num = classes_num

        self.D = config.D

        self.loss_all = 0.0

        input_channel = 1

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(2, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (5, words_dim), padding=(2, 0))

        self.dropout1 = nn.Dropout(config.dropout)

        self.capsulize = routing(D=self.ks*self.output_channel,
                                 n_in=1,
                                 n_out=self.classes_num,
                                 in_dim=self.D,
                                 out_dim=self.D,
                                 device=device)

        self.dropout2 = nn.Dropout(config.dropout)

        self.classifier = nn.Linear(self.D, 1)

    def cnn(self, x):
        x = x.unsqueeze(1)

        x = [F.relu(self.conv1(x)).squeeze(3),
             F.relu(self.conv2(x)).squeeze(3),
             F.relu(self.conv3(x)).squeeze(3)]

        x = [x[0], x[1][:, :, 0:-1], x[2]]

        x = T.cat(x, dim=1)
        x = x.permute(0, 2, 1).contiguous()

        return x

    def forward(self, x, mask):

        N, S = x.size()
        max_len = 600

        if S > max_len:
            x = x[:, 0:max_len]
            S = max_len
            mask = mask[:, 0:S]

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        x = F.embedding(x, dropped_embeddings, padding_idx=self.pad_idx)
        x = self.cnn(x)

        x = x.view(N, S, self.ks*self.output_channel)
        x = self.dropout1(x)

        capsule_out = self.capsulize(x, mask)

        #capsule_out = self.dropout2(capsule_out)
        #logit = capsule_out.norm(dim=-1)
        logit = T.sigmoid(self.classifier(capsule_out).view(N, self.classes_num))

        return logit
