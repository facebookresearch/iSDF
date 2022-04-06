# Copyright (c) Meta Platforms, Inc. and affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import numpy as np

from isdf import geometry


def scale_input(tensor, transform=None, scale=None):
    if transform is not None:
        t_shape = tensor.shape
        tensor = geometry.transform.transform_3D_grid(
            tensor.view(-1, 3), transform=transform)
        tensor = tensor.view(t_shape)

    if scale is not None:
        tensor = tensor * scale

    return tensor


class PostionalEncoding(torch.nn.Module):
    def __init__(
        self,
        min_deg=0,
        max_deg=6,
        scale=0.1,
        transform=None,
    ):
        super(PostionalEncoding, self).__init__()
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.n_freqs = max_deg - min_deg + 1
        self.scale = scale
        self.transform = transform

        self.dirs = torch.tensor([
            0.8506508, 0, 0.5257311,
            0.809017, 0.5, 0.309017,
            0.5257311, 0.8506508, 0,
            1, 0, 0,
            0.809017, 0.5, -0.309017,
            0.8506508, 0, -0.5257311,
            0.309017, 0.809017, -0.5,
            0, 0.5257311, -0.8506508,
            0.5, 0.309017, -0.809017,
            0, 1, 0,
            -0.5257311, 0.8506508, 0,
            -0.309017, 0.809017, -0.5,
            0, 0.5257311, 0.8506508,
            -0.309017, 0.809017, 0.5,
            0.309017, 0.809017, 0.5,
            0.5, 0.309017, 0.809017,
            0.5, -0.309017, 0.809017,
            0, 0, 1,
            -0.5, 0.309017, 0.809017,
            -0.809017, 0.5, 0.309017,
            -0.809017, 0.5, -0.309017
        ]).reshape(-1, 3).T

        frequency_bands = 2.0 ** np.linspace(
            self.min_deg, self.max_deg, self.n_freqs)
        self.embedding_size = 2 * self.dirs.shape[1] * self.n_freqs + 3

        print(
            "Icosahedron embedding with periods:",
            (2 * np.pi) / (frequency_bands * self.scale),
            " -- embedding size:", self.embedding_size
        )

    def vis_embedding(self):
        x = torch.linspace(0, 5, 640)
        embd = x * self.scale
        if self.gauss_embed:
            frequency_bands = torch.norm(self.B_layer.weight, dim=1)
            frequency_bands = torch.sort(frequency_bands)[0]
        else:
            frequency_bands = 2.0 ** torch.linspace(
                self.min_deg, self.max_deg, self.n_freqs)

        embd = embd[..., None] * frequency_bands
        embd = torch.sin(embd)

        import matplotlib.pylab as plt
        plt.imshow(embd.T, cmap='hot', interpolation='nearest',
                   aspect='auto', extent=[0, 5, 0, embd.shape[1]])
        plt.colorbar()
        plt.xlabel("x values")
        plt.ylabel("embedings")
        plt.show()

    def forward(self, tensor):
        frequency_bands = 2.0 ** torch.linspace(
            self.min_deg, self.max_deg, self.n_freqs,
            dtype=tensor.dtype, device=tensor.device)

        tensor = scale_input(
            tensor, transform=self.transform, scale=self.scale)

        proj = torch.matmul(tensor, self.dirs.to(tensor.device))
        xb = torch.reshape(
            proj[..., None] * frequency_bands,
            list(proj.shape[:-1]) + [-1]
        )
        embedding = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
        embedding = torch.cat([tensor] + [embedding], dim=-1)

        return embedding
