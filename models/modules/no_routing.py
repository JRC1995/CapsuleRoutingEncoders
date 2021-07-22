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

        self.Wcap = nn.Parameter(T.randn(n_in, D, in_dim).float().to(device))
        self.Bcap = nn.Parameter(T.zeros(n_in, in_dim).float().to(device))
        nn.init.xavier_uniform_(self.Wcap.data)

        self.bij = T.zeros(1, self.n_in, self.n_out, 1).float().to(device)
        self.leak = T.zeros(1, 1, 1, 1).float().to(device)

        self.Wvotes = nn.Parameter(T.randn(n_in, n_out, in_dim, out_dim))
        self.Bvotes = nn.Parameter(T.zeros(n_in, n_out, 1, out_dim))

        nn.init.xavier_uniform_(self.Wvotes.data)

    def squash(self, x, dim=-1):
        norm_x = x.norm(dim=dim, keepdim=True)

        f1 = norm_x.pow(2)/(norm_x.pow(2)+1)
        f2 = x/(norm_x+self.epsilon)

        return f1*f2

    def forward(self, x, mask):

        N, n_in, D = x.size()

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

        vj = T.sum(votes_ij, dim=1)
        vj = F.gelu(vj)

        return vj
