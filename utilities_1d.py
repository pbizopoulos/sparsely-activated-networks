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


def identity_1d(x, kernel_size):
    return x

def relu_1d(x, kernel_size):
    return torch.relu(x)

def save_images_1d(model, dataset_name, data, xlim_weights, path_results):
    fig, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_signal.pdf', bbox_inches='tight')
    plt.close()

    model.eval()
    with torch.no_grad():
        reconstructed, activations_list = model(data.unsqueeze(0).unsqueeze(0))
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
                if model.sparse_activation.__name__ == 'identity_1d':
                    sparse_activation_name = 'Identity'
                elif model.sparse_activation.__name__ == 'relu_1d':
                    sparse_activation_name = 'ReLU'
                elif model.sparse_activation.__name__ == 'topk_absolutes_1d':
                    sparse_activation_name = 'top-k absolutes'
                elif model.sparse_activation.__name__ == 'extrema_pool_indices_1d':
                    sparse_activation_name = 'Extrema-Pool idx'
                elif model.sparse_activation.__name__ == 'extrema_1d':
                    sparse_activation_name = 'Extrema'
                plt.ylabel(sparse_activation_name, fontsize=20)
            if model.sparse_activation.__name__ == 'relu_1d':
                plt.title(dataset_name, fontsize=20)
            plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_kernel_{index_weights}.pdf', bbox_inches='tight')
            plt.close()

            similarity = _conv1d_same_padding(data.unsqueeze(0).unsqueeze(0), weights)[0, 0]
            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(similarity.cpu().detach().numpy(), 'g')
            plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_similarity_{index_weights}.pdf', bbox_inches='tight')
            plt.close()

            fig, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            p = activations.nonzero()[:, 0]
            plt.plot(similarity.cpu().detach().numpy(), 'g', alpha=0.5)
            if p.shape[0] != 0:
                plt.stem(p.cpu().detach().numpy(), activations[p.cpu().detach().numpy()].cpu().detach().numpy(), 'c', basefmt=' ', use_line_collection=True)
            plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_activations_{index_weights}.pdf', bbox_inches='tight')
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
            plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_reconstruction_{index_weights}.pdf', bbox_inches='tight')
            plt.close()

        fig, ax = plt.subplots()
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(data.cpu().detach().numpy(), alpha=0.5)
        plt.plot(reconstructed[0, 0].cpu().detach().numpy(), 'r')
        plt.ylim([data.min(), data.max()])
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_reconstructed.pdf', bbox_inches='tight')
        plt.close()

def download_physionet(dataset_name_list, path_cache):
    for dataset_name in dataset_name_list:
        path = f'{path_cache}/{dataset_name}'
        if not os.path.exists(path):
            record_name = wfdb.get_record_list(dataset_name)[0]
            wfdb.dl_database(dataset_name, path, records=[record_name], annotators=None)


class PhysionetDataset(Dataset):
    def __init__(self, training_validation_test, dataset_name, path_cache):
        files = glob.glob(f'{path_cache}/{dataset_name}/*.hea')
        file = files[0]
        filename = os.path.splitext(os.path.basename(file))[0]
        records = wfdb.rdrecord(f'{path_cache}/{dataset_name}/{filename}')
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


def download_uci_epilepsy(path_cache):
    path = f'{path_cache}/UCI-epilepsy'
    if not os.path.exists(path):
        os.mkdir(path)
        urllib.request.urlretrieve('http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv', f'{path}/data.csv')

class UCIepilepsyDataset(Dataset):
    def __init__(self, training_validation_test, path_cache):
        dataset = pd.read_csv(f'{path_cache}/UCI-epilepsy/data.csv')
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
