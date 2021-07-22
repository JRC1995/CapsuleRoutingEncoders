import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math


class routing(nn.Module):
    def __init__(self, D, n_in, n_out, in_dim, out_dim, device):
        super(routing, self).__init__()

        self.D = D
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.n_in = n_in
        self.n_out = n_out
        self.epsilon = 1.0e-8

        self.pad_inf = T.tensor(float("-inf")).to(device)
        self.zeros = T.tensor(0.0).to(device)

        self.Wcap = nn.Parameter(T.randn(n_in, D, in_dim).float().to(device))
        self.Bcap = nn.Parameter(T.zeros(n_in, in_dim).float().to(device))
        nn.init.xavier_uniform_(self.Wcap.data)

        self.bij = T.zeros(1, self.n_in, self.n_out, 1).float().to(device)

        self.Wvotes = nn.Parameter(T.randn(n_in, n_out, in_dim, out_dim))
        self.Bvotes = nn.Parameter(T.zeros(n_in, n_out, 1, out_dim))

        nn.init.xavier_uniform_(self.Wvotes.data)

    def forward(self, x, mask, iters=3):

        N, n_in, D = x.size()

        attention_mask = T.where(mask == float(0),
                                 self.pad_inf,
                                 self.zeros).view(N, n_in, 1, 1)

        Wcap = self.Wcap.view(1, self.n_in, D, self.in_dim)
        Bcap = self.Bcap.view(1, self.n_in, self.in_dim)
        x = F.gelu(T.matmul(x, Wcap) + Bcap)

        x = x.view(N, n_in, self.in_dim)
        x = x.view(N, n_in, 1, self.in_dim)

        if self.n_in == 1:
            Wvotes = self.Wvotes.repeat(n_in, 1, 1, 1)
        else:
            Wvotes = self.Wvotes
        votes_ij = T.einsum('ijdh,...icd->...ijch', Wvotes, x) + self.Bvotes

        in_caps = n_in
        out_caps = self.n_out
        votes_ij = votes_ij.view(N, in_caps, out_caps, self.out_dim)

        mask = mask.view(N, in_caps, 1, 1)
        votes_ij = votes_ij*mask

        if self.n_in == 1:
            bij = self.bij.repeat(N, in_caps, 1, 1)
        else:
            bij = self.bij.repeat(N, 1, 1, 1)

        for i in range(iters):
            aij = F.softmax(bij+attention_mask, dim=1)
            vj = T.tanh(T.sum(aij*votes_ij, dim=1))

            if i != iters-1:
                votes_ij_ = votes_ij.view(N, in_caps, out_caps, 1, self.out_dim)
                vj_ = vj.view(N, 1, out_caps, self.out_dim, 1)
                bij = bij + T.matmul(votes_ij_, vj_).view(N, in_caps, out_caps, 1)

        vj = vj.view(N, out_caps, self.out_dim)

        return vj
