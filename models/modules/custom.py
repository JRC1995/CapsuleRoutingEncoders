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

        self.score = nn.Linear(out_dim, 1)
        self.alpha_score = nn.Linear(out_dim, 1)

        self.fe1 = nn.Linear(self.out_dim, self.out_dim)
        self.fe2 = nn.Linear(self.out_dim, self.out_dim)
        self.fn1 = nn.Linear(self.out_dim, self.out_dim)
        self.fn2 = nn.Linear(self.out_dim, self.out_dim)

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

        fn_votes_ij = self.fn1(votes_ij)
        fe_votes_ij = self.fe1(votes_ij)

        bij = self.score(votes_ij)
        aij = F.softmax(bij*mask + attention_mask, dim=1)

        for i in range(iters):

            if i != 0:
                old_vj = vj.clone()
                new_vj = F.gelu(T.sum(aij*votes_ij, dim=1))
                alpha = T.sigmoid(self.alpha_score(new_vj))
                vj = alpha*new_vj + (1-alpha)*old_vj
            else:
                vj = F.gelu(T.sum(aij*votes_ij, dim=1))

            if i != iters-1:

                fe_votes_ij_ = fe_votes_ij.view(N, in_caps, out_caps, self.out_dim)
                fn_votes_ij_ = fn_votes_ij.view(N, in_caps, out_caps, self.out_dim)

                vj_ = vj.view(N, 1, out_caps, self.out_dim)

                #alpha = T.sigmoid(self.alpha_score(vj_))

                E = T.sum(fe_votes_ij_*self.fe2(vj_), dim=-1, keepdim=True)
                M = -T.sum(T.abs(fn_votes_ij_-self.fn2(vj_)), dim=-1, keepdim=True)
                #E = E - T.mean(E, dim=1, keepdim=True)
                #M = M - T.mean(M, dim=1, keepdim=True)

                aij = T.tanh(E)*T.sigmoid(M*mask+attention_mask)  # alpha*aij + (1-alpha)*

        vj = vj.view(N, out_caps, self.out_dim)

        return vj
