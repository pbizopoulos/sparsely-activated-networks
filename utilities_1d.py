import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import urllib
import wfdb

from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset

from sparsely_activated_networks_pytorch import _conv1d_same_padding

plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'

class Hook():
    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, module, input, output):
        self.input = input
        self.output = output


class identity_1d(nn.Module):
    def __init__(self, k):
        super().__init__()

    def forward(self, input):
        return input

class relu_1d(nn.Module):
    def __init__(self, k):
        super().__init__()

    def forward(self, input):
        return F.relu(input)


def save_images_1d(model, sparse_activation_name, dataset_name, data, xlim_weights, results_dir):
    model = model.to('cpu')
    fig, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-signal')
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
            fig, ax = plt.subplots(figsize=(2, 2.2))
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_visible(False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(weights.cpu().detach().numpy(), 'r')
            plt.xlim([0, xlim_weights])
            if dataset_name == 'apnea-ecg':
                if sparse_activation_name == 'identity_1d':
                    sparse_activation_name = 'Identity'
                elif sparse_activation_name == 'relu_1d':
                    sparse_activation_name = 'ReLU'
                elif sparse_activation_name == 'topk_absolutes_1d':
                    sparse_activation_name = 'top-k absolutes'
                elif sparse_activation_name == 'extrema_pool_indices_1d':
                    sparse_activation_name = 'Extrema-Pool idx'
                elif sparse_activation_name == 'extrema_1d':
                    sparse_activation_name = 'Extrema'
                plt.ylabel(sparse_activation_name, fontsize=20)
            if sparse_activation_name == 'relu_1d':
                plt.title(dataset_name, fontsize=20)
            plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-kernel-{index_weights}')
            plt.close()

            similarity = _conv1d_same_padding(data.unsqueeze(0).unsqueeze(0), weights)[0, 0]
            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(similarity.cpu().detach().numpy(), 'g')
            plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-similarity-{index_weights}')
            plt.close()

            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            p = torch.nonzero(activations, as_tuple=False)[:, 0]
            plt.plot(similarity.cpu().detach().numpy(), 'g', alpha=0.5)
            if p.shape[0] != 0:
                plt.stem(p.cpu().detach().numpy(), activations[p.cpu().detach().numpy()].cpu().detach().numpy(), 'c', basefmt=' ', use_line_collection=True)
            plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-activations-{index_weights}')
            plt.close()

            reconstruction = _conv1d_same_padding(activations.unsqueeze(0).unsqueeze(0), weights)[0, 0]
            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            reconstruction = reconstruction.cpu().detach().numpy()
            left = p - weights.shape[0]/2
            right = p + weights.shape[0]/2
            if weights.shape[0] % 2 == 1:
                right += 1
            step = np.zeros_like(reconstruction, dtype=np.bool)
            left[left < 0] = 0
            right[right > reconstruction.shape[0]] = reconstruction.shape[0]
            for l, r in zip(left, right):
                step[int(l):int(r)] = True
            pos_signal = reconstruction.copy()
            neg_signal = reconstruction.copy()
            pos_signal[step] = np.nan
            neg_signal[~step] = np.nan
            plt.plot(pos_signal)
            plt.plot(neg_signal, color='r')
            plt.ylim([data.min(), data.max()])
            plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-reconstruction-{index_weights}')
            plt.close()

        fig, ax = plt.subplots()
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(data.cpu().detach().numpy(), alpha=0.5)
        plt.plot(reconstructed[0, 0].cpu().detach().numpy(), 'r')
        plt.ylim([data.min(), data.max()])
        plt.savefig(f'{results_dir}/{dataset_name}-{sparse_activation_name.replace("_", "-")}-{len(model.weights_list)}-reconstructed')
        plt.close()

def download_physionet(dataset_name_list, cache_dir):
    for dataset_name in dataset_name_list:
        path = f'{cache_dir}/{dataset_name}'
        if not os.path.exists(path):
            record_name = wfdb.get_record_list(dataset_name)[0]
            wfdb.dl_database(dataset_name, path, records=[record_name], annotators=None)


class PhysionetDataset(Dataset):
    def __init__(self, training_validation_test, dataset_name, cache_dir):
        files = glob.glob(f'{cache_dir}/{dataset_name}/*.hea')
        file = files[0]
        filename = os.path.splitext(os.path.basename(file))[0]
        records = wfdb.rdrecord(f'{cache_dir}/{dataset_name}/{filename}')
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


def download_uci_epilepsy(cache_dir):
    path = f'{cache_dir}/UCI-epilepsy'
    if not os.path.exists(path):
        os.mkdir(path)
        urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv', f'{path}/data.csv')

class UCIepilepsyDataset(Dataset):
    def __init__(self, training_validation_test, cache_dir):
        dataset = pd.read_csv(f'{cache_dir}/UCI-epilepsy/data.csv')
        dataset['y'].loc[dataset['y'] == 3] = 2
        dataset['y'].loc[dataset['y'] == 5] = 3
        dataset['y'].loc[dataset['y'] == 4] = 3
        data_all = dataset.drop(columns=['Unnamed: 0', 'y'])
        data_max = data_all.max().max()
        data_min = data_all.min().min()
        data_all = 2*(data_all - data_min)/(data_max - data_min) - 1
        labels_all = dataset['y']
        last_training_index = int(data_all.shape[0]*0.76)
        last_validation_index = int(data_all.shape[0]*0.88)
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
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool1d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out
