import torch

from torch import nn
from torch.nn import functional as F


class SAN2d(nn.Module):
    def __init__(self, sparse_activation, kernel_size_list, sparsity_density_list, device):
        super(SAN2d, self).__init__()
        self.device = device
        self.sparse_activation = sparse_activation
        self.kernel_size_list = kernel_size_list
        self.sparsity_density_list = sparsity_density_list
        self.neuron_list = nn.ModuleList()
        for kernel_size, sparsity_density in zip(kernel_size_list, sparsity_density_list):
            self.neuron_list.append(Neuron2d(sparse_activation, kernel_size, sparsity_density, device))

    def forward(self, batch_x):
        reconstructions = torch.zeros(batch_x.shape[0], len(self.neuron_list), *batch_x.shape[1:]).to(self.device)
        similarity_list = torch.zeros_like(reconstructions).to(self.device)
        activations_list = torch.zeros_like(reconstructions).to(self.device)
        for index_neuron, neuron in enumerate(self.neuron_list):
            reconstructions[:, index_neuron], similarity_list[:, index_neuron], activations_list[:, index_neuron] = neuron(batch_x)
        reconstructed = torch.sum(reconstructions, dim=1)
        return reconstructed, similarity_list, activations_list, reconstructions


class Neuron2d(nn.Module):
    def __init__(self, sparse_activation, kernel_size, sparsity_density, device):
        super(Neuron2d, self).__init__()
        self.sparse_activation = sparse_activation
        self.kernel_size = kernel_size
        self.weights = nn.Parameter(torch.empty(kernel_size, kernel_size, device=device))
        nn.init.normal_(self.weights, 0, 0.1)
        self.sparsity_density = sparsity_density

    def forward(self, x):
        similarity = conv2d_same_padding(x, self.weights)
        extrema = self.sparse_activation(similarity, self.sparsity_density)
        reconstruction = conv2d_same_padding(extrema, self.weights)
        return reconstruction, similarity, extrema


def conv2d_same_padding(x, weights):
    padding = weights.shape[0] - 1
    odd = int(padding % 2 != 0)
    if odd:
        x = F.pad(x, [0, odd, 0, odd])
    out = F.conv2d(x, weights.unsqueeze(0).unsqueeze(0), padding=padding//2)
    return out

def identity_2d(x, kernel_size):
    return x

def relu_2d(x, kernel_size):
    return torch.relu(x)

def topk_absolutes_2d(x, kernel_size):
    k = int(x.shape[-1]/kernel_size) ** 2
    x_flattened = x.view(x.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    _, extrema_indices = torch.topk(abs(x_flattened), k)
    for x_, e_i, p_ in zip(x_flattened, extrema_indices, primary_extrema):
        p_[e_i] = x_[e_i]
    return primary_extrema.view(x.shape)

def extrema_pool_indices_2d(x, kernel_size):
    x_flattened = x.view(x.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    _, extrema_indices = F.max_pool2d_with_indices(abs(x), kernel_size)
    for x_, e_i, p_ in zip(x_flattened, extrema_indices, primary_extrema):
        p_[e_i.view(-1)] = x_[e_i.view(-1)]
    return primary_extrema.view(x.shape)

def extrema_2d(x, minimum_extrema_distance):
    primary_extrema = torch.zeros_like(x)
    dx = x[:, :, :, 1:] - x[:, :, :, :-1]
    dy = x[:, :, 1:, :] - x[:, :, :-1, :]
    dx_padright_greater = F.pad(dx, [0, 1, 0, 0]) > 0
    dx_padleft_less = F.pad(dx, [1, 0, 0, 0]) <= 0
    dy_padright_greater = F.pad(dy, [0, 0, 0, 1]) > 0
    dy_padleft_less = F.pad(dy, [0, 0, 1, 0]) <= 0
    sign = (1 - torch.sign(x)).bool()
    valleys_x = dx_padright_greater & dx_padleft_less & sign
    valleys_y = dy_padright_greater & dy_padleft_less & sign
    peaks_x = ~dx_padright_greater & ~dx_padleft_less & ~sign
    peaks_y = ~dy_padright_greater & ~dy_padleft_less & ~sign
    peaks = peaks_x & peaks_y
    valleys = valleys_x & valleys_y
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for x_, e_, p_ in zip(x, extrema, primary_extrema):
        x_.squeeze_(0)
        extrema_indices = e_.nonzero()
        extrema_indices_indices = torch.argsort(abs(x_[e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices]
        is_secondary_extrema = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for i, (extrema_index_x, extrema_index_y) in enumerate(extrema_indices_sorted):
            if not is_secondary_extrema[i]:
                extrema_indices_r = extrema_indices_sorted[:, 0] >= extrema_index_x - minimum_extrema_distance
                extrema_indices_l = extrema_indices_sorted[:, 0] <= extrema_index_x + minimum_extrema_distance
                extrema_indices_t = extrema_indices_sorted[:, 1] >= extrema_index_y - minimum_extrema_distance
                extrema_indices_b = extrema_indices_sorted[:, 1] <= extrema_index_y + minimum_extrema_distance
                extrema_indices_m = extrema_indices_r & extrema_indices_l & extrema_indices_t & extrema_indices_b
                is_secondary_extrema = is_secondary_extrema | extrema_indices_m
                is_secondary_extrema[i] = False
        primary_extrema_indices = extrema_indices_sorted[~is_secondary_extrema]
        for primary_extrema_index in primary_extrema_indices:
            p_[:, primary_extrema_index[0], primary_extrema_index[1]] = x_[primary_extrema_index[0], primary_extrema_index[1]]
    return primary_extrema
