import torch

from torch import nn
from torch.nn import functional as F


class SAN1d(nn.Module):
    def __init__(self, sparse_activation, kernel_size_list, sparsity_density_list, device):
        super(SAN1d, self).__init__()
        self.device = device
        self.sparse_activation = sparse_activation
        self.kernel_size_list = kernel_size_list
        self.sparsity_density_list = sparsity_density_list
        self.neuron_list = nn.ModuleList()
        for kernel_size, sparsity_density in zip(kernel_size_list, sparsity_density_list):
            self.neuron_list.append(Neuron1d(sparse_activation, kernel_size, sparsity_density, device))

    def forward(self, batch_x):
        reconstructions = torch.zeros(batch_x.shape[0], len(self.neuron_list), *batch_x.shape[1:]).to(self.device)
        similarity_list = torch.zeros_like(reconstructions).to(self.device)
        activations_list = torch.zeros_like(reconstructions).to(self.device)
        for index_neuron, neuron in enumerate(self.neuron_list):
            reconstructions[:, index_neuron], similarity_list[:, index_neuron], activations_list[:, index_neuron] = neuron(batch_x)
        reconstructed = torch.sum(reconstructions, dim=1)
        return reconstructed, similarity_list, activations_list, reconstructions


class Neuron1d(nn.Module):
    def __init__(self, sparse_activation, kernel_size, sparsity_density, device):
        super(Neuron1d, self).__init__()
        self.sparse_activation = sparse_activation
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size, device=device))
        nn.init.normal_(self.weights, 0, 0.1)
        self.sparsity_density = sparsity_density

    def forward(self, x):
        similarity = conv1d_same_padding(x, self.weights)
        extrema = self.sparse_activation(similarity, self.sparsity_density)
        reconstruction = conv1d_same_padding(extrema, self.weights)
        return reconstruction, similarity, extrema


def conv1d_same_padding(x, weights):
    padding = weights.shape[0] - 1
    odd = int(padding % 2 != 0)
    if odd:
        x = F.pad(x, [0, odd])
    out = F.conv1d(x, weights.unsqueeze(0).unsqueeze(0), padding=padding//2)
    return out

def identity_1d(x, kernel_size):
    return x

def relu_1d(x, kernel_size):
    return torch.relu(x)
