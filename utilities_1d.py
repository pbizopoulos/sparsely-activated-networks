import glob
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch
import urllib
import wfdb

from torch.utils.data import Dataset


def save_images_1d(model, dataset_name, data, xlim_weights, device, path_results):
    fig, ax = plt.subplots(constrained_layout=True)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_signal.pdf')
    plt.close()

    model.eval()
    reconstructed, similarity_list, activations_list, reconstructions = model(data.unsqueeze(0).unsqueeze(0).to(device))
    for index_neuron, (neuron, similarity, activations, reconstruction) in enumerate(zip(model.neuron_list, similarity_list[0, :, 0], activations_list[0, :, 0], reconstructions[0, :, 0])):
        fig, ax = plt.subplots(constrained_layout=True, figsize=(2, 2.2))
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(neuron.weights.flip(0).cpu().detach().numpy().T, 'r')
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
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_kernel_{index_neuron}.pdf')
        plt.close()

        fig, ax = plt.subplots(constrained_layout=True)
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(similarity.cpu().detach().numpy(), 'g')
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_similarity_{index_neuron}.pdf')
        plt.close()

        fig, ax = plt.subplots(constrained_layout=True)
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        p = activations.nonzero()[:, 0]
        plt.plot(similarity.cpu().detach().numpy(), 'g', alpha=0.5)
        if p.shape[0] != 0:
            plt.stem(p.cpu().detach().numpy(), activations[p.cpu().detach().numpy()].cpu().detach().numpy(), 'c', basefmt=' ', use_line_collection=True)
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_activations_{index_neuron}.pdf')
        plt.close()

        fig, ax = plt.subplots(constrained_layout=True)
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        reconstruction = reconstruction.cpu().detach().numpy()
        left = p - neuron.weights.shape[0]/2
        right = p + neuron.weights.shape[0]/2
        if neuron.weights.shape[0] % 2 == 1:
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
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_reconstruction_{index_neuron}.pdf')
        plt.close()

    fig, ax = plt.subplots(constrained_layout=True)
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy(), alpha=0.5)
    plt.plot(reconstructed[0, 0].cpu().detach().numpy(), 'r')
    plt.ylim([data.min(), data.max()])
    plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.neuron_list)}_reconstructed.pdf')
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
