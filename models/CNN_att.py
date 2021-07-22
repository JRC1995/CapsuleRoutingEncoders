# Adapted from: https://github.com/castorini/hedwig/tree/master/models/kim_cnn

import torch as T
import torch.nn as nn
import torch.nn.functional as F


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

        self.pad_inf = T.tensor(float("-inf")).to(device)
        self.zeros = T.tensor(0.0).to(device)

        self.loss_all = 0.0

        input_channel = 1

        self.conv1 = nn.Conv2d(input_channel, self.output_channel, (3, words_dim), padding=(1, 0))
        self.conv2 = nn.Conv2d(input_channel, self.output_channel, (4, words_dim), padding=(2, 0))
        self.conv3 = nn.Conv2d(input_channel, self.output_channel, (5, words_dim), padding=(2, 0))

        self.att1 = nn.Linear(self.ks*self.output_channel, config.att_dim)
        self.att2 = nn.Linear(config.att_dim, 1)

        self.out_dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(self.ks*self.output_channel, classes_num)

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

        if S > 600:
            x = x[:, 0:600]
            S = 600
            mask = mask[:, 0:S]

        attention_mask = T.where(mask == float(0),
                                 self.pad_inf,
                                 self.zeros)

        embeddings_dropout_mask = self.embedding_dropout(self.embedding_ones)
        dropped_embeddings = self.embeddings*embeddings_dropout_mask

        x = F.embedding(x, dropped_embeddings, padding_idx=self.pad_idx)
        x = self.cnn(x)

        e = self.att2(T.tanh(self.att1(x)))

        a = F.softmax(e.view(N, S)+attention_mask, dim=-1).unsqueeze(-1)
        x = x.view(N, S, -1)
        v = T.sum(a*x, dim=1)

        v = self.out_dropout(v)
        logit = T.sigmoid(self.classifier(v))
        return logit
