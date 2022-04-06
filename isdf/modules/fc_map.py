# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import grad


def gradient(inputs, outputs):
    d_points = torch.ones_like(
        outputs, requires_grad=False, device=outputs.device)
    points_grad = grad(
        outputs=outputs,
        inputs=inputs,
        grad_outputs=d_points,
        create_graph=True,
        retain_graph=True,
        only_inputs=True)[0]
    return points_grad


def chunks(
    pc,
    chunk_size,
    fc_sdf_map,
    to_cpu=False,
):
    n_pts = pc.shape[0]
    n_chunks = int(np.ceil(n_pts / chunk_size))
    alphas = []
    for n in range(n_chunks):
        start = n * chunk_size
        end = start + chunk_size
        chunk = pc[start:end, :]

        alpha = fc_sdf_map(chunk)

        alpha = alpha.squeeze(dim=-1)
        if to_cpu:
            alpha = alpha.cpu()
        alphas.append(alpha)

    alphas = torch.cat(alphas, dim=-1)

    return alphas


def fc_block(in_f, out_f):
    return torch.nn.Sequential(
        torch.nn.Linear(in_f, out_f),
        torch.nn.Softplus(beta=100)
    )


def init_weights(m, init_fn=torch.nn.init.xavier_normal_):
    if isinstance(m, torch.nn.Linear):
        init_fn(m.weight)


class SDFMap(nn.Module):
    def __init__(
        self,
        positional_encoding,
        hidden_size=256,
        hidden_layers_block=1,
        scale_output=1.,
    ):
        super(SDFMap, self).__init__()
        self.scale_output = scale_output

        self.positional_encoding = positional_encoding
        embedding_size = self.positional_encoding.embedding_size

        self.in_layer = fc_block(embedding_size, hidden_size)

        hidden1 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid1 = torch.nn.Sequential(*hidden1)

        self.cat_layer = fc_block(
            hidden_size + embedding_size, hidden_size)

        hidden2 = [fc_block(hidden_size, hidden_size)
                   for _ in range(hidden_layers_block)]
        self.mid2 = torch.nn.Sequential(*hidden2)

        self.out_alpha = torch.nn.Linear(hidden_size, 1)

        self.apply(init_weights)

    def forward(self, x, noise_std=None, pe_mask=None, sdf1=None):
        x_pe = self.positional_encoding(x)
        if pe_mask is not None:
            x_pe = torch.mul(x_pe, pe_mask)

        fc1 = self.in_layer(x_pe)
        fc2 = self.mid1(fc1)
        fc2_x = torch.cat((fc2, x_pe), dim=-1)
        fc3 = self.cat_layer(fc2_x)
        fc4 = self.mid2(fc3)
        raw = self.out_alpha(fc4)

        if noise_std is not None:
            noise = torch.randn(raw.shape, device=x.device) * noise_std
            raw = raw + noise
        alpha = raw * self.scale_output

        return alpha.squeeze(-1)
