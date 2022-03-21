import glob
import os
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import torch
import wfdb
from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor

artifacts_dir = os.getenv('ARTIFACTSDIR')
full = os.getenv('FULL')
plt.rcParams['font.size'] = 20
plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['savefig.format'] = 'pdf'


class Identity1D(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return input_


class Relu1D(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return F.relu(input_)


def topk_absolutes_1d(input_, k):
    primary_extrema = torch.zeros_like(input_)
    _, extrema_indices = torch.topk(abs(input_), k)
    return primary_extrema.scatter(-1, extrema_indices, input_.gather(-1, extrema_indices))


class TopKAbsolutes1D(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input_):
        return topk_absolutes_1d(input_, self.k)


def extrema_pool_indices_1d(input_, kernel_size):
    primary_extrema = torch.zeros_like(input_)
    _, extrema_indices = F.max_pool1d_with_indices(abs(input_), kernel_size)
    return primary_extrema.scatter(-1, extrema_indices, input_.gather(-1, extrema_indices))


class ExtremaPoolIndices1D(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input_):
        return extrema_pool_indices_1d(input_, self.k)


def extrema_1d(input_, minimum_extrema_distance):
    primary_extrema = torch.zeros_like(input_)
    dx = input_[:, :, 1:] - input_[:, :, :-1]
    dx_padright_greater = F.pad(dx, [0, 1]) > 0
    dx_padleft_less = F.pad(dx, [1, 0]) <= 0
    sign = (1 - torch.sign(input_)).bool()
    valleys = dx_padright_greater & dx_padleft_less & sign
    peaks = ~dx_padright_greater & ~dx_padleft_less & ~sign
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for index, (x_, e_) in enumerate(zip(input_, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices][:, 0]
        is_secondary_extrema = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for i, extrema_index in enumerate(extrema_indices_sorted):
            if not is_secondary_extrema[i]:
                extrema_indices_r = extrema_indices_sorted >= extrema_index - minimum_extrema_distance
                extrema_indices_l = extrema_indices_sorted <= extrema_index + minimum_extrema_distance
                extrema_indices_m = extrema_indices_r & extrema_indices_l
                is_secondary_extrema = is_secondary_extrema | extrema_indices_m
                is_secondary_extrema[i] = False
        primary_extrema_indices = extrema_indices_sorted[~is_secondary_extrema]
        primary_extrema[index, :, primary_extrema_indices] = x_[0, primary_extrema_indices]
    return primary_extrema


class Extrema1D(nn.Module):
    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_):
        return extrema_1d(input_, self.minimum_extrema_distance)


class SAN1d(nn.Module):
    def __init__(self, sparse_activation_list, kernel_size_list):
        super().__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1 * torch.ones(kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        reconstructions_sum = torch.zeros_like(batch_x)
        for sparse_activation, weights in zip(self.sparse_activation_list, self.weights_list):
            similarity = F.conv1d(batch_x, weights.unsqueeze(0).unsqueeze(0), padding='same')
            activations_list = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + F.conv1d(activations_list, weights.unsqueeze(0).unsqueeze(0), padding='same')
        return reconstructions_sum


class Identity2D(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return input_


class Relu2D(nn.Module):
    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return F.relu(input_)


def topk_absolutes_2d(input_, k):
    x_flattened = input_.view(input_.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    _, extrema_indices = torch.topk(abs(x_flattened), k)
    return primary_extrema.scatter(-1, extrema_indices, x_flattened.gather(-1, extrema_indices)).view(input_.shape)


class TopKAbsolutes2D(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input_):
        return topk_absolutes_2d(input_, self.k)


def extrema_pool_indices_2d(input_, kernel_size):
    x_flattened = input_.view(input_.shape[0], -1)
    primary_extrema = torch.zeros_like(x_flattened)
    _, extrema_indices = F.max_pool2d_with_indices(abs(input_), kernel_size)
    return primary_extrema.scatter(-1, extrema_indices[..., 0, 0], x_flattened.gather(-1, extrema_indices[..., 0, 0])).view(input_.shape)


class ExtremaPoolIndices2D(nn.Module):
    def __init__(self, k):
        super().__init__()
        self.k = k

    def forward(self, input_):
        return extrema_pool_indices_2d(input_, self.k)


def extrema_2d(input_, minimum_extrema_distance):
    primary_extrema = torch.zeros_like(input_)
    dx = input_[:, :, :, 1:] - input_[:, :, :, :-1]
    dy = input_[:, :, 1:, :] - input_[:, :, :-1, :]
    dx_padright_greater = F.pad(dx, [0, 1, 0, 0]) > 0
    dx_padleft_less = F.pad(dx, [1, 0, 0, 0]) <= 0
    dy_padright_greater = F.pad(dy, [0, 0, 0, 1]) > 0
    dy_padleft_less = F.pad(dy, [0, 0, 1, 0]) <= 0
    sign = (1 - torch.sign(input_)).bool()
    valleys_x = dx_padright_greater & dx_padleft_less & sign
    valleys_y = dy_padright_greater & dy_padleft_less & sign
    peaks_x = ~dx_padright_greater & ~dx_padleft_less & ~sign
    peaks_y = ~dy_padright_greater & ~dy_padleft_less & ~sign
    peaks = peaks_x & peaks_y
    valleys = valleys_x & valleys_y
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for index, (x_, e_) in enumerate(zip(input_, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices]
        is_secondary_extrema = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for i, (extrema_index_x, extrema_index_y) in enumerate(extrema_indices_sorted):
            if not is_secondary_extrema[i]:
                extrema_indices_r = extrema_indices_sorted[:, 0] >= extrema_index_x - minimum_extrema_distance[0]
                extrema_indices_l = extrema_indices_sorted[:, 0] <= extrema_index_x + minimum_extrema_distance[0]
                extrema_indices_t = extrema_indices_sorted[:, 1] >= extrema_index_y - minimum_extrema_distance[1]
                extrema_indices_b = extrema_indices_sorted[:, 1] <= extrema_index_y + minimum_extrema_distance[1]
                extrema_indices_m = extrema_indices_r & extrema_indices_l & extrema_indices_t & extrema_indices_b
                is_secondary_extrema = is_secondary_extrema | extrema_indices_m
                is_secondary_extrema[i] = False
        primary_extrema_indices = extrema_indices_sorted[~is_secondary_extrema]
        for primary_extrema_index in primary_extrema_indices:
            primary_extrema[index, :, primary_extrema_index[0], primary_extrema_index[1]] = x_[0, primary_extrema_index[0], primary_extrema_index[1]]
    return primary_extrema


class Extrema2D(nn.Module):
    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_):
        return extrema_2d(input_, self.minimum_extrema_distance)


class SAN2d(nn.Module):
    def __init__(self, sparse_activation_list, kernel_size_list):
        super().__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1 * torch.ones(kernel_size, kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        reconstructions_sum = torch.zeros_like(batch_x)
        for index_weights, (sparse_activation, weights) in enumerate(zip(self.sparse_activation_list, self.weights_list)):
            similarity = F.conv2d(batch_x, weights.unsqueeze(0).unsqueeze(0), padding='same')
            activations_list = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + F.conv2d(activations_list, weights.unsqueeze(0).unsqueeze(0), padding='same')
        return reconstructions_sum


class Hook:
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, _, input_, output):
        self.input_ = input_
        self.output = output


def save_images_2d(model, sparse_activation_name, data, dataset_name):
    model = model.to('cpu')
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data.cpu().detach().numpy(), cmap='twilight', vmin=-2, vmax=2)
    plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-signal'))
    plt.close()
    model.eval()
    hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_list = []
        for hook_handle in hook_handle_list:
            activations_list.append(hook_handle.output)
        activations_list = torch.stack(activations_list, 1)
        for index_weights, (weights, activations) in enumerate(zip(model.weights_list, activations_list[0, :, 0])):
            plt.figure(figsize=(4.8 / 2, 4.8 / 2))
            plt.imshow(weights.flip(0).flip(1).cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(weights).max(), vmax=2 * abs(weights).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-kernel-{index_weights}'))
            plt.close()
            similarity = F.conv2d(data.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(similarity.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(similarity).max(), vmax=2 * abs(similarity).max())
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-similarity-{index_weights}'))
            plt.close()
            plt.figure()
            plt.imshow(activations.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(activations).max(), vmax=2 * abs(activations).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-activations-{index_weights}'))
            plt.close()
            reconstruction = F.conv2d(activations.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            plt.figure()
            plt.imshow(reconstruction.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(reconstruction).max(), vmax=2 * abs(reconstruction).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-reconstruction-{index_weights}'))
            plt.close()
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(reconstructed).max(), vmax=2 * abs(reconstructed).max())
        plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-reconstructed'))
        plt.close()


class FNN(nn.Module):
    def __init__(self, sample_data, num_classes):
        super().__init__()
        self.fc = nn.Linear(sample_data.shape[-1] * sample_data.shape[-2], num_classes)

    def forward(self, batch_x):
        x = batch_x.view(batch_x.shape[0], -1)
        out = self.fc(x)
        return out


def save_images_1d(model, sparse_activation_name, dataset_name, data, xlim_weights):
    model = model.to('cpu')
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-signal'))
    plt.close()
    model.eval()
    hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_list = []
        for hook_handle in hook_handle_list:
            activations_list.append(hook_handle.output)
        activations_list = torch.stack(activations_list, 1)
        for index_weights, (weights, activations) in enumerate(zip(model.weights_list, activations_list[0, :, 0])):
            _, ax = plt.subplots(figsize=(2, 2.2))
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_visible(False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(weights.cpu().detach().numpy(), 'r')
            plt.xlim([0, xlim_weights])
            if dataset_name == 'apnea-ecg':
                plt.ylabel(sparse_activation_name, fontsize=20)
            if sparse_activation_name == 'relu':
                plt.title(dataset_name, fontsize=20)
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-kernel-{index_weights}'))
            plt.close()
            similarity = F.conv1d(data.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(similarity.cpu().detach().numpy(), 'g')
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-similarity-{index_weights}'))
            plt.close()
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            p = torch.nonzero(activations, as_tuple=False)[:, 0]
            plt.plot(similarity.cpu().detach().numpy(), 'g', alpha=0.5)
            if p.shape[0] != 0:
                plt.stem(p.cpu().detach().numpy(), activations[p.cpu().detach().numpy()].cpu().detach().numpy(), linefmt='c', basefmt=' ')
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-activations-{index_weights}'))
            plt.close()
            reconstruction = F.conv1d(activations.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            reconstruction = reconstruction.cpu().detach().numpy()
            lefts = p - weights.shape[0] / 2
            rights = p + weights.shape[0] / 2
            if weights.shape[0] % 2 == 1:
                rights += 1
            step = np.zeros_like(reconstruction, dtype=bool)
            lefts[lefts < 0] = 0
            rights[rights > reconstruction.shape[0]] = reconstruction.shape[0]
            for left, right in zip(lefts, rights):
                step[int(left):int(right)] = True
            pos_signal = reconstruction.copy()
            neg_signal = reconstruction.copy()
            pos_signal[step] = np.nan
            neg_signal[~step] = np.nan
            plt.plot(pos_signal)
            plt.plot(neg_signal, color='r')
            plt.ylim([data.min(), data.max()])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-reconstruction-{index_weights}'))
            plt.close()
        _, ax = plt.subplots()
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(data.cpu().detach().numpy(), alpha=0.5)
        plt.plot(reconstructed[0, 0].cpu().detach().numpy(), 'r')
        plt.ylim([data.min(), data.max()])
        plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-reconstructed'))
        plt.close()


class PhysionetDataset(Dataset):
    def __init__(self, training_validation_test, dataset_name):
        dataset_dir = join(artifacts_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            record_name = wfdb.get_record_list(dataset_name)[0]
            wfdb.dl_database(dataset_name, dataset_dir, records=[record_name], annotators=None)
        files = glob.glob(join(dataset_dir, '*.hea'))
        filename = os.path.splitext(os.path.basename(files[0]))[0]
        records = wfdb.rdrecord(join(dataset_dir, filename))
        data = torch.tensor(records.p_signal[:12000, 0], dtype=torch.float)
        if training_validation_test == 'training':
            self.data = data[:6000]
        elif training_validation_test == 'validation':
            self.data = data[6000:8000]
        elif training_validation_test == 'test':
            self.data = data[8000:]
        self.data = self.data.reshape((-1, 1, 1000))

    def __getitem__(self, index):
        d = self.data[index] - self.data[index].mean()
        d /= d.std()
        return (d, 0)

    def __len__(self):
        return self.data.shape[0]


class UCIepilepsyDataset(Dataset):
    def __init__(self, path, training_validation_test):
        if not os.path.exists(path):
            os.mkdir(path)
            with open(join(path, 'data.csv'), 'wb') as file:
                response = requests.get('https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv')
                file.write(response.content)
        dataset = pd.read_csv(join(path, 'data.csv'))
        dataset['y'].replace(3, 2, inplace=True)
        dataset['y'].replace(4, 3, inplace=True)
        dataset['y'].replace(5, 3, inplace=True)
        data_all = dataset.drop(columns=['Unnamed: 0', 'y'])
        data_max = data_all.max().max()
        data_min = data_all.min().min()
        data_all = 2 * (data_all - data_min) / (data_max - data_min) - 1
        labels_all = dataset['y']
        last_training_index = int(data_all.shape[0] * 0.76)
        last_validation_index = int(data_all.shape[0] * 0.88)
        if training_validation_test == 'training':
            self.data = torch.tensor(data_all.values[:last_training_index, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[:last_training_index].values) - 1
        elif training_validation_test == 'validation':
            self.data = torch.tensor(data_all.values[last_training_index:last_validation_index, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[last_training_index:last_validation_index].values) - 1
        elif training_validation_test == 'test':
            self.data = torch.tensor(data_all.values[last_validation_index:, :], dtype=torch.float)
            self.labels = torch.tensor(labels_all[last_validation_index:].values) - 1
        self.data.unsqueeze_(1)

    def __getitem__(self, index):
        return (self.data[index], self.labels[index])

    def __len__(self):
        return self.labels.shape[0]


class CNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, input_):
        out = F.relu(self.conv1(input_))
        out = F.max_pool1d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out


def calculate_inverse_compression_ratio(model, data, num_activations):
    activation_multiplier = 1 + len(model.weights_list[0].shape)
    num_parameters = sum(weights.shape[0] for weights in model.weights_list)
    return (activation_multiplier * num_activations + num_parameters) / (data.shape[-1] * data.shape[-2])


def train_unsupervised_model(model, optimizer, training_dataloader):
    device = next(model.parameters()).device
    model.train()
    for data, _ in training_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        data_reconstructed = model(data)
        reconstruction_loss = F.l1_loss(data, data_reconstructed)
        reconstruction_loss.backward()
        optimizer.step()


def validate_or_test_unsupervised_model(model, hook_handle_list, dataloader):
    device = next(model.parameters()).device
    model.eval()
    num_activations = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    with torch.no_grad():
        for index, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data_reconstructed = model(data)
            activations_list = []
            for hook_handle in hook_handle_list:
                activations_list.append(hook_handle.output)
            activations_list = torch.stack(activations_list, 1)
            reconstruction_loss[index] = F.l1_loss(data, data_reconstructed) / F.l1_loss(data, torch.zeros_like(data))
            num_activations[index] = torch.nonzero(activations_list, as_tuple=False).shape[0]
    inverse_compression_ratio = calculate_inverse_compression_ratio(model, data, num_activations)
    flithos = np.mean([np.sqrt(i ** 2 + r ** 2) for i, r in zip(inverse_compression_ratio, reconstruction_loss)])
    return flithos, inverse_compression_ratio, reconstruction_loss


def train_supervised_model(supervised_model, unsupervised_model, optimizer, training_dataloader):
    device = next(supervised_model.parameters()).device
    supervised_model.train()
    for data, target in training_dataloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        data_reconstructed = unsupervised_model(data)
        output = supervised_model(data_reconstructed)
        classification_loss = F.cross_entropy(output, target)
        classification_loss.backward()
        optimizer.step()


def validate_or_test_supervised_model(supervised_model, unsupervised_model, hook_handle_list, dataloader):
    device = next(supervised_model.parameters()).device
    supervised_model.eval()
    correct = 0
    num_activations = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    with torch.no_grad():
        for index, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            data_reconstructed = unsupervised_model(data)
            activations_list = []
            for hook_handle in hook_handle_list:
                activations_list.append(hook_handle.output)
            activations_list = torch.stack(activations_list, 1)
            reconstruction_loss[index] = F.l1_loss(data, data_reconstructed) / F.l1_loss(data, torch.zeros_like(data))
            num_activations[index] = torch.nonzero(activations_list, as_tuple=False).shape[0]
            output = supervised_model(data_reconstructed)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    inverse_compression_ratio = calculate_inverse_compression_ratio(unsupervised_model, data, num_activations)
    flithos = np.mean([np.sqrt(i ** 2 + r ** 2) for i, r in zip(inverse_compression_ratio, reconstruction_loss)])
    return flithos, inverse_compression_ratio, reconstruction_loss, 100 * correct / len(dataloader.sampler)


def main():
    num_epochs_physionet = 30
    num_epochs = 5
    physionet_kernel_size_list_list_range = range(1, 250)
    uci_epilepsy_training_range = range(8740)
    uci_epilepsy_validation_range = range(1380)
    uci_epilepsy_test_range = range(1380)
    mnist_fashionmnist_training_range_list = [range(50000), range(50000)]
    mnist_fashionmnist_validation_range_list = [range(50000, 60000), range(50000, 60000)]
    mnist_fashionmnist_test_range_list = [range(10000), range(10000)]
    if not full:
        num_epochs_physionet = 3
        num_epochs = 2
        physionet_kernel_size_list_list_range = range(1, 10)
        uci_epilepsy_training_range = range(10)
        uci_epilepsy_validation_range = range(10)
        uci_epilepsy_test_range = range(10)
        mnist_fashionmnist_training_range_list = [range(10), range(10)]
        mnist_fashionmnist_validation_range_list = [range(10, 20), range(10, 20)]
        mnist_fashionmnist_test_range_list = [range(10), range(10)]
    np.random.seed(0)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    sparse_activation_name_list = ['Identity', 'ReLU', 'top-k absolutes', 'Extrema-Pool indices', 'Extrema']
    uci_epilepsy_kernel_size_range = range(8, 16)
    mnist_fashionmnist_kernel_size_range_list = [range(1, 7), range(1, 7)]
    print('Physionet, X: mean reconstruction loss, Y: mean inverse compression ratio, Color: sparse activation')
    dataset_name_list = ['apnea-ecg', 'bidmc', 'bpssrat', 'cebsdb', 'ctu-uhb-ctgdb', 'drivedb', 'emgdb', 'mitdb', 'noneeg', 'prcp', 'shhpsgdb', 'slpdb', 'sufhsdb', 'voiced', 'wrist']
    xlim_weights_list = [74, 113, 10, 71, 45, 20, 9, 229, 37, 105, 15, 232, 40, 70, 173]
    sparse_activation_list = [Identity1D, Relu1D, TopKAbsolutes1D, ExtremaPoolIndices1D, Extrema1D]
    kernel_size_list_list = [[k] for k in physionet_kernel_size_list_list_range]
    batch_size = 2
    lr = 0.01
    sparse_activation_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    physionet_latex_table = []
    mean_flithos = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list)))
    flithos_all_validation = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list), num_epochs_physionet))
    kernel_size_list_best = np.zeros((len(sparse_activation_list), len(dataset_name_list)))
    for_density_plot = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list), 2))
    for index_dataset_name, (dataset_name, xlim_weights) in enumerate(zip(dataset_name_list, xlim_weights_list)):
        print(dataset_name)
        physionet_latex_table_row = []
        training_dataset = PhysionetDataset('training', dataset_name)
        training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, shuffle=True)
        validation_dataset = PhysionetDataset('validation', dataset_name)
        validation_dataloader = DataLoader(dataset=validation_dataset)
        test_dataset = PhysionetDataset('test', dataset_name)
        test_dataloader = DataLoader(dataset=test_dataset)
        fig, ax_main = plt.subplots(constrained_layout=True, figsize=(6, 6))
        for index_sparse_activation, (sparse_activation, sparse_activation_color, sparse_activation_name) in enumerate(zip(sparse_activation_list, sparse_activation_color_list, sparse_activation_name_list)):
            mean_flithos_best = float('inf')
            for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
                mean_flithos_epoch_best = float('inf')
                if sparse_activation == TopKAbsolutes1D:
                    sparsity_density_list = [int(test_dataset.data.shape[-1] / k) for k in kernel_size_list]
                elif sparse_activation == Extrema1D:
                    sparsity_density_list = np.clip([k - 3 for k in kernel_size_list], 1, 999).tolist()
                else:
                    sparsity_density_list = kernel_size_list
                sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
                model = SAN1d(sparse_activation_list_, kernel_size_list).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
                for epoch in range(num_epochs_physionet):
                    train_unsupervised_model(model, optimizer, training_dataloader)
                    flithos_epoch, *_ = validate_or_test_unsupervised_model(model, hook_handle_list, validation_dataloader)
                    flithos_all_validation[index_sparse_activation, index_dataset_name, index_kernel_size_list, epoch] = flithos_epoch.mean()
                    if flithos_epoch.mean() < mean_flithos_epoch_best:
                        model_epoch_best = model
                        mean_flithos_epoch_best = flithos_epoch.mean()
                flithos_epoch_best, inverse_compression_ratio_epoch_best, reconstruction_loss_epoch_best = validate_or_test_unsupervised_model(model_epoch_best, hook_handle_list, test_dataloader)
                mean_flithos[index_sparse_activation, index_dataset_name, index_kernel_size_list] = flithos_epoch_best.mean()
                plt.sca(ax_main)
                plt.plot(reconstruction_loss_epoch_best.mean(), inverse_compression_ratio_epoch_best.mean(), 'o', c=sparse_activation_color, markersize=3)
                for_density_plot[index_sparse_activation, index_dataset_name, index_kernel_size_list] = [reconstruction_loss_epoch_best.mean(), inverse_compression_ratio_epoch_best.mean()]
                if mean_flithos[index_sparse_activation, index_dataset_name, index_kernel_size_list] < mean_flithos_best:
                    kernel_size_list_best[index_sparse_activation, index_dataset_name] = kernel_size_list[0]
                    inverse_compression_ratio_best = inverse_compression_ratio_epoch_best
                    reconstruction_loss_best = reconstruction_loss_epoch_best
                    mean_flithos_best = mean_flithos[index_sparse_activation, index_dataset_name, index_kernel_size_list]
                    model_best = model_epoch_best
            physionet_latex_table_row.extend([kernel_size_list_best[index_sparse_activation, index_dataset_name], inverse_compression_ratio_best.mean(), reconstruction_loss_best.mean(), mean_flithos_best])
            save_images_1d(model_best, sparse_activation_name.lower().replace(' ', '-'), dataset_name, test_dataset[0][0][0], xlim_weights)
            ax_main.arrow(reconstruction_loss_best.mean(), inverse_compression_ratio_best.mean(), 1.83 - reconstruction_loss_best.mean(), 2.25 - 0.5 * index_sparse_activation - inverse_compression_ratio_best.mean())
            fig.add_axes([0.75, 0.81 - 0.165 * index_sparse_activation, 0.1, 0.1], facecolor='y')
            plt.plot(model_best.weights_list[0].flip(0).cpu().detach().numpy().T, c=sparse_activation_color)
            plt.xlim([0, xlim_weights])
            plt.xticks([])
            plt.yticks([])
        physionet_latex_table.append(physionet_latex_table_row)
        plt.sca(ax_main)
        plt.xlim([0, 2.5])
        plt.ylim([0, 2.5])
        plt.xlabel(r'$\tilde{\mathcal{L}}$')
        plt.ylabel(r'$CR^{-1}$')
        plt.grid(True)
        plt.title(dataset_name)
        plt.axhspan(2, 2.5, alpha=0.3, color='r')
        plt.axhspan(1, 2, alpha=0.3, color='orange')
        plt.axvspan(1, 2.5, alpha=0.3, color='gray')
        wedge = patches.Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
        ax_main.add_patch(wedge)
        plt.savefig(join(artifacts_dir, dataset_name))
        plt.close()
    header = ['$m$', r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$']
    columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    df = pd.DataFrame(physionet_latex_table, columns=columns, index=dataset_name_list)
    df.index.names = ['Datasets']
    styler = df.style
    styler.format(precision=2, formatter={columns[0]: '{:.0f}', columns[4]: '{:.0f}', columns[8]: '{:.0f}', columns[12]: '{:.0f}', columns[16]: '{:.0f}'})
    styler.to_latex(join(artifacts_dir, 'table-flithos-variable-kernel-size.tex'), hrules=True, multicol_align='c')
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    var = np.zeros((len(dataset_name_list), num_epochs_physionet))
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for index, (sparse_activation, sparse_activation_name, sparse_activation_color, kernel_size_best, c) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, kernel_size_list_best, flithos_all_validation)):
        t = np.arange(1, c.shape[-1] + 1)
        for j, (c_, k_) in enumerate(zip(c, kernel_size_best)):
            var[j] = c_[int(k_ - 1)]
        mu = var.mean(axis=0)
        sigma = var.std(axis=0)
        ax.fill_between(t, mu + sigma, mu - sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, fontsize=12, loc='lower left')
    plt.xlabel(r'epochs')
    plt.ylabel(r'$\bar\varphi$')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 2.5])
    plt.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(join(artifacts_dir, 'mean-flithos-validation-epochs'))
    plt.close()
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for index, (sparse_activation, sparse_activation_name, sparse_activation_color, c) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, mean_flithos)):
        t = np.arange(1, c.shape[1] + 1)
        mu = c.mean(axis=0)
        sigma = c.std(axis=0)
        ax.fill_between(t, mu + sigma, mu - sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, fontsize=12, loc='lower right')
    plt.xlabel(r'$m$')
    plt.ylabel(r'$\bar\varphi$')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 2.5])
    plt.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(join(artifacts_dir, 'mean-flithos-variable-kernel-size-list'))
    plt.close()
    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    fig_legend = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2), range(2), range(2))
    patch_non_sparse_model_description = patches.Patch(color='r', alpha=0.3, label='non-sparse model description')
    patch_worse_cr_than_original_data = patches.Patch(color='orange', alpha=0.3, label=r'worse $CR^{-1}$ than original data')
    patch_worse_l_than_constant_prediction = patches.Patch(color='gray', alpha=0.3, label=r'worse $\tilde{\mathcal{L}}$ than constant prediction')
    patch_varphi_less_than_one = patches.Patch(color='g', alpha=0.3, label=r'$\bar\varphi < 1$')
    line2d_list = []
    for sparse_activation_name, sparse_activation_color in zip(sparse_activation_name_list, sparse_activation_color_list):
        line2d_list.append(Line2D([0], [0], marker='o', color='w', label=sparse_activation_name, markerfacecolor=sparse_activation_color))
    legend_elements = [patch_non_sparse_model_description, patch_worse_cr_than_original_data, patch_worse_l_than_constant_prediction, patch_varphi_less_than_one, line2d_list[0], line2d_list[1], line2d_list[2], line2d_list[3], line2d_list[4]]
    fig_legend.legend(handles=legend_elements, fontsize=22, loc='upper center')
    plt.savefig(join(artifacts_dir, 'legend'))
    plt.close()
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    for_density_plot = for_density_plot.reshape(for_density_plot.shape[0], -1, 2)
    nbins = 200
    yi, xi = np.mgrid[0:2.5:nbins * 1j, 0:2.5:nbins * 1j]
    for sparse_activation_color, c in zip(sparse_activation_color_list, for_density_plot):
        k = gaussian_kde(c.T)
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
        plt.contourf(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
    plt.axhspan(2, 2.5, alpha=0.3, color='r')
    plt.axhspan(1, 2, alpha=0.3, color='orange')
    plt.axvspan(1, 2.5, alpha=0.3, color='gray')
    wedge = patches.Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
    ax.add_patch(wedge)
    plt.xlabel(r'$\tilde{\mathcal{L}}$')
    plt.ylabel(r'$CR^{-1}$')
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.grid(True)
    plt.savefig(join(artifacts_dir, 'crrl-density-plot'))
    plt.close()
    print('UCI baseline, Supervised CNN classification')
    batch_size = 64
    lr = 0.01
    uci_download_path = join(artifacts_dir, 'UCI-epilepsy')
    training_dataset = UCIepilepsyDataset(uci_download_path, 'training')
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    validation_dataset = UCIepilepsyDataset(uci_download_path, 'validation')
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    test_dataset = UCIepilepsyDataset(uci_download_path, 'test')
    test_dataloader = DataLoader(dataset=test_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_test_range))
    best_accuracy = 0
    supervised_model = CNN(len(training_dataset.labels.unique())).to(device)
    optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
    for epoch in range(num_epochs):
        supervised_model.train()
        for data, target in training_dataloader:
            data = data.to(device)
            target = target.to(device)
            optimizer.zero_grad()
            output = supervised_model(data)
            classification_loss = F.cross_entropy(output, target)
            classification_loss.backward()
            optimizer.step()
        supervised_model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for data, target in validation_dataloader:
                data = data.to(device)
                target = target.to(device)
                output = supervised_model(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += output.shape[0]
        accuracy = 100 * correct / total
        if best_accuracy < accuracy:
            supervised_model_best = supervised_model
            best_accuracy = accuracy
    supervised_model.eval()
    with torch.no_grad():
        for data, target in test_dataloader:
            data = data.to(device)
            target = target.to(device)
            output = supervised_model_best(data)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
            total += output.shape[0]
    uci_epilepsy_supervised_accuracy = 100 * correct / total
    print('UCI-epilepsy, Supervised CNN classification')
    dataset_name = 'UCI-epilepsy'
    sparse_activation_list = [Identity1D, Relu1D, TopKAbsolutes1D, ExtremaPoolIndices1D, Extrema1D]
    kernel_size_list_list = [2 * [k] for k in uci_epilepsy_kernel_size_range]
    batch_size = 64
    lr = 0.01
    supervised_latex_table = []
    training_dataset = UCIepilepsyDataset(uci_download_path, 'training')
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    validation_dataset = UCIepilepsyDataset(uci_download_path, 'validation')
    validation_dataloader = DataLoader(dataset=validation_dataset, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    test_dataset = UCIepilepsyDataset(uci_download_path, 'test')
    test_dataloader = DataLoader(dataset=test_dataset, sampler=SubsetRandomSampler(uci_epilepsy_test_range))
    for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
        print(f'index_kernel_size_list: {index_kernel_size_list}')
        supervised_latex_table_row = []
        for index_sparse_activation, (sparse_activation, sparse_activation_name) in enumerate(zip(sparse_activation_list, sparse_activation_name_list)):
            if sparse_activation == TopKAbsolutes1D:
                sparsity_density_list = [int(test_dataset.data.shape[-1] / k) for k in kernel_size_list]
            elif sparse_activation == Extrema1D:
                sparsity_density_list = np.clip([k - 2 for k in kernel_size_list], 1, 999).tolist()
            else:
                sparsity_density_list = kernel_size_list
            sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
            model = SAN1d(sparse_activation_list_, kernel_size_list).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
            mean_flithos_epoch_best = float('inf')
            for epoch in range(num_epochs):
                train_unsupervised_model(model, optimizer, training_dataloader)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model, hook_handle_list, validation_dataloader)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    model_epoch_best = model
                    mean_flithos_epoch_best = flithos_epoch.mean()
            for weights in model.weights_list:
                weights.requires_grad_(False)
            mean_flithos_epoch_best = float('inf')
            supervised_model = CNN(len(training_dataset.labels.unique())).to(device).to(device)
            optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_supervised_model(supervised_model, model_epoch_best, optimizer, training_dataloader)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model_epoch_best, hook_handle_list, validation_dataloader)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    supervised_model_best = supervised_model
                    model_best = model_epoch_best
                    mean_flithos_epoch_best = flithos_epoch.mean()
            flithos, inverse_compression_ratio, reconstruction_loss, accuracy = validate_or_test_supervised_model(supervised_model_best, model_best, hook_handle_list, test_dataloader)
            supervised_latex_table_row.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - uci_epilepsy_supervised_accuracy])
            if kernel_size_list[0] == 10:
                save_images_1d(model_best, sparse_activation_name.lower().replace(' ', '-'), dataset_name, test_dataset[0][0][0], kernel_size_list[0])
        supervised_latex_table.append(supervised_latex_table_row)
    header = [r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$', r'A\textsubscript{$\pm$\%}']
    columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    df = pd.DataFrame(supervised_latex_table, columns=columns, index=uci_epilepsy_kernel_size_range)
    df.index.names = [r'$m$']
    styler = df.style
    styler.format(precision=2, formatter={columns[3]: '{:.1f}', columns[7]: '{:.1f}', columns[11]: '{:.1f}', columns[15]: '{:.1f}', columns[19]: '{:.1f}'})
    styler.to_latex(join(artifacts_dir, 'table-uci-epilepsy-supervised.tex'), hrules=True, multicol_align='c')
    dataset_name_list = ['MNIST', 'FashionMNIST']
    dataset_list = [MNIST, FashionMNIST]
    mnist_fashionmnist_supervised_accuracy_list = [0, 0]
    for index_dataset_name, (dataset_name, dataset, mnist_fashionmnist_kernel_size_range, mnist_fashionmnist_training_range, mnist_fashionmnist_validation_range, mnist_fashionmnist_test_range) in enumerate(zip(dataset_name_list, dataset_list, mnist_fashionmnist_kernel_size_range_list, mnist_fashionmnist_training_range_list, mnist_fashionmnist_validation_range_list, mnist_fashionmnist_test_range_list)):
        print(f'{dataset_name} baseline, Supervised FNN classification')
        batch_size = 64
        lr = 0.01
        training_validation_dataset = dataset(artifacts_dir, download=True, train=True, transform=ToTensor())
        training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_fashionmnist_training_range))
        validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range), batch_size=batch_size)
        test_dataset = dataset(artifacts_dir, train=False, transform=ToTensor())
        test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(mnist_fashionmnist_test_range))
        best_accuracy = 0
        supervised_model = FNN(training_validation_dataset.data[0], len(training_validation_dataset.classes)).to(device)
        optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
        for epoch in range(num_epochs):
            supervised_model.train()
            for data, target in training_dataloader:
                data = data.to(device)
                target = target.to(device)
                optimizer.zero_grad()
                output = supervised_model(data)
                classification_loss = F.cross_entropy(output, target)
                classification_loss.backward()
                optimizer.step()
            supervised_model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for data, target in validation_dataloader:
                    data = data.to(device)
                    target = target.to(device)
                    output = supervised_model(data)
                    pred = output.argmax(dim=1)
                    correct += (pred == target).sum().item()
                    total += output.shape[0]
            accuracy = 100 * correct / total
            if best_accuracy < accuracy:
                supervised_model_best = supervised_model
                best_accuracy = accuracy
        supervised_model.eval()
        with torch.no_grad():
            for data, target in test_dataloader:
                data = data.to(device)
                target = target.to(device)
                output = supervised_model_best(data)
                pred = output.argmax(dim=1)
                correct += (pred == target).sum().item()
                total += output.shape[0]
        mnist_fashionmnist_supervised_accuracy_list[index_dataset_name] = 100 * correct / total
        print(f'{dataset_name}, Supervised FNN classification')
        sparse_activation_list = [Identity2D, Relu2D, TopKAbsolutes2D, ExtremaPoolIndices2D, Extrema2D]
        kernel_size_list_list = [2 * [k] for k in mnist_fashionmnist_kernel_size_range]
        batch_size = 64
        lr = 0.01
        supervised_latex_table = []
        training_validation_dataset = dataset(artifacts_dir, download=True, train=True, transform=ToTensor())
        training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_fashionmnist_training_range))
        validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range))
        test_dataset = dataset(artifacts_dir, train=False, transform=ToTensor())
        test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(mnist_fashionmnist_test_range))
        for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
            print(f'index_kernel_size_list: {index_kernel_size_list}')
            supervised_latex_table_row = []
            for index_sparse_activation, (sparse_activation, sparse_activation_name) in enumerate(zip(sparse_activation_list, sparse_activation_name_list)):
                if sparse_activation == TopKAbsolutes2D:
                    sparsity_density_list = [int(test_dataset.data.shape[-1] / k) ** 2 for k in kernel_size_list]
                elif sparse_activation == Extrema2D:
                    sparsity_density_list = np.clip([k - 2 for k in kernel_size_list], 1, 999).tolist()
                    sparsity_density_list = [[s, s] for s in sparsity_density_list]
                else:
                    sparsity_density_list = kernel_size_list
                sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
                model = SAN2d(sparse_activation_list_, kernel_size_list).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
                mean_flithos_epoch_best = float('inf')
                for epoch in range(num_epochs):
                    train_unsupervised_model(model, optimizer, training_dataloader)
                    flithos_epoch, *_ = validate_or_test_unsupervised_model(model, hook_handle_list, validation_dataloader)
                    if flithos_epoch.mean() < mean_flithos_epoch_best:
                        model_epoch_best = model
                        mean_flithos_epoch_best = flithos_epoch.mean()
                for weights in model.weights_list:
                    weights.requires_grad_(False)
                mean_flithos_epoch_best = float('inf')
                supervised_model = FNN(training_validation_dataset.data[0], len(training_validation_dataset.classes)).to(device)
                optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
                for epoch in range(num_epochs):
                    train_supervised_model(supervised_model, model_epoch_best, optimizer, training_dataloader)
                    flithos_epoch, *_ = validate_or_test_unsupervised_model(model_epoch_best, hook_handle_list, validation_dataloader)
                    if flithos_epoch.mean() < mean_flithos_epoch_best:
                        supervised_model_best = supervised_model
                        model_best = model_epoch_best
                        mean_flithos_epoch_best = flithos_epoch.mean()
                flithos, inverse_compression_ratio, reconstruction_loss, accuracy = validate_or_test_supervised_model(supervised_model_best, model_best, hook_handle_list, test_dataloader)
                supervised_latex_table_row.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - mnist_fashionmnist_supervised_accuracy_list[index_dataset_name]])
                if kernel_size_list[0] == 4:
                    save_images_2d(model_best, sparse_activation_name.lower().replace(' ', '-'), test_dataset[0][0][0], dataset_name)
            supervised_latex_table.append(supervised_latex_table_row)
        header = [r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$', r'A\textsubscript{$\pm$\%}']
        columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
        df = pd.DataFrame(supervised_latex_table, columns=columns, index=mnist_fashionmnist_kernel_size_range)
        df.index.names = [r'$m$']
        styler = df.style
        styler.format(precision=2, formatter={columns[3]: '{:.1f}', columns[7]: '{:.1f}', columns[11]: '{:.1f}', columns[15]: '{:.1f}', columns[19]: '{:.1f}'})
        styler.to_latex(join(artifacts_dir, f'table-{dataset_name.lower()}-supervised.tex'), hrules=True, multicol_align='c')
    df = pd.DataFrame({'key': ['uci-epilepsy-supervised-accuracy', 'mnist-supervised-accuracy', 'fashionmnist-supervised-accuracy'], 'value': [uci_epilepsy_supervised_accuracy, mnist_fashionmnist_supervised_accuracy_list[0], mnist_fashionmnist_supervised_accuracy_list[1]]})
    df.to_csv(join(artifacts_dir, 'keys-values.csv'), index=False, float_format='%.2f')


if __name__ == '__main__':
    main()
