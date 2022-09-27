from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from matplotlib.ticker import MaxNLocator
from os import environ
from os.path import join
from scipy.stats import gaussian_kde
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import FashionMNIST, MNIST
from torchvision.transforms import ToTensor
import glob
import numpy as np
import os
import pandas as pd
import requests
import torch
import wfdb


class CNN(nn.Module):

    def __init__(self, classes_num):
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)

    def forward(self, input_):
        out = functional.relu(self.conv1(input_))
        out = functional.max_pool1d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        out = self.fc3(out)
        return out


class Extrema1D(nn.Module):

    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_):
        return extrema_1d(input_, self.minimum_extrema_distance)


class Extrema2D(nn.Module):

    def __init__(self, minimum_extrema_distance):
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_):
        return extrema_2d(input_, self.minimum_extrema_distance)


class ExtremaPoolIndices1D(nn.Module):

    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input_):
        return extrema_pool_indices_1d(input_, self.pool_size)


class ExtremaPoolIndices2D(nn.Module):

    def __init__(self, pool_size):
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input_):
        return extrema_pool_indices_2d(input_, self.pool_size)


class FNN(nn.Module):

    def __init__(self, classes_num, sample_data):
        super().__init__()
        self.fc = nn.Linear(sample_data.shape[-1] * sample_data.shape[-2], classes_num)

    def forward(self, batch_x):
        out = batch_x.view(batch_x.shape[0], -1)
        out = self.fc(out)
        return out


class Hook:

    def __init__(self, module):
        self.hook = module.register_forward_hook(self.hook_fn)

    def hook_fn(self, _, input_, output):
        self.input_ = input_
        self.output = output


class Identity1D(nn.Module):

    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return input_


class Identity2D(nn.Module):

    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return input_


class PhysionetDataset(Dataset):

    def __getitem__(self, index):
        out = self.data[index] - self.data[index].mean()
        out /= out.std()
        return (out, 0)

    def __init__(self, artifacts_dir, dataset_name, training_validation_test):
        dataset_dir = join(artifacts_dir, dataset_name)
        if not os.path.exists(dataset_dir):
            record_name = wfdb.get_record_list(dataset_name)[0]
            wfdb.dl_database(dataset_name, dataset_dir, records=[record_name], annotators=None)
        files = glob.glob(join(dataset_dir, '*.hea'))
        file_name = os.path.splitext(os.path.basename(files[0]))[0]
        records = wfdb.rdrecord(join(dataset_dir, file_name))
        data = torch.tensor(records.p_signal[:12000, 0], dtype=torch.float)
        if training_validation_test == 'training':
            self.data = data[:6000]
        elif training_validation_test == 'validation':
            self.data = data[6000:8000]
        elif training_validation_test == 'test':
            self.data = data[8000:]
        self.data = self.data.reshape((-1, 1, 1000))

    def __len__(self):
        return self.data.shape[0]


class Relu1D(nn.Module):

    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return functional.relu(input_)


class Relu2D(nn.Module):

    def __init__(self, _):
        super().__init__()

    def forward(self, input_):
        return functional.relu(input_)


class SAN1d(nn.Module):

    def __init__(self, kernel_size_list, sparse_activation_list):
        super().__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1 * torch.ones(kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        reconstructions_sum = torch.zeros_like(batch_x)
        for (sparse_activation, weights) in zip(self.sparse_activation_list, self.weights_list):
            similarity = functional.conv1d(batch_x, weights.unsqueeze(0).unsqueeze(0), padding='same')
            activations_list = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + functional.conv1d(activations_list, weights.unsqueeze(0).unsqueeze(0), padding='same')
        return reconstructions_sum


class SAN2d(nn.Module):

    def __init__(self, kernel_size_list, sparse_activation_list):
        super().__init__()
        self.sparse_activation_list = nn.ModuleList(sparse_activation_list)
        self.weights_list = nn.ParameterList([nn.Parameter(0.1 * torch.ones(kernel_size, kernel_size)) for kernel_size in kernel_size_list])

    def forward(self, batch_x):
        reconstructions_sum = torch.zeros_like(batch_x)
        for (sparse_activation, weights) in zip(self.sparse_activation_list, self.weights_list):
            similarity = functional.conv2d(batch_x, weights.unsqueeze(0).unsqueeze(0), padding='same')
            activations_list = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + functional.conv2d(activations_list, weights.unsqueeze(0).unsqueeze(0), padding='same')
        return reconstructions_sum


class TopKAbsolutes1D(nn.Module):

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, input_):
        return topk_absolutes_1d(input_, self.topk)


class TopKAbsolutes2D(nn.Module):

    def __init__(self, topk):
        super().__init__()
        self.topk = topk

    def forward(self, input_):
        return topk_absolutes_2d(input_, self.topk)


class UCIepilepsyDataset(Dataset):

    def __getitem__(self, index):
        return (self.data[index], self.classes[index])

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
        classes_all = dataset['y']
        last_training_index = int(data_all.shape[0] * 0.76)
        last_validation_index = int(data_all.shape[0] * 0.88)
        if training_validation_test == 'training':
            self.data = torch.tensor(data_all.values[:last_training_index, :], dtype=torch.float)
            self.classes = torch.tensor(classes_all[:last_training_index].values) - 1
        elif training_validation_test == 'validation':
            self.data = torch.tensor(data_all.values[last_training_index:last_validation_index, :], dtype=torch.float)
            self.classes = torch.tensor(classes_all[last_training_index:last_validation_index].values) - 1
        elif training_validation_test == 'test':
            self.data = torch.tensor(data_all.values[last_validation_index:, :], dtype=torch.float)
            self.classes = torch.tensor(classes_all[last_validation_index:].values) - 1
        self.data.unsqueeze_(1)

    def __len__(self):
        return self.classes.shape[0]


def calculate_inverse_compression_ratio(activations_num, data, model):
    activation_multiplier = 1 + len(model.weights_list[0].shape)
    parameters_num = sum((weights.shape[0] for weights in model.weights_list))
    return (activation_multiplier * activations_num + parameters_num) / (data.shape[-1] * data.shape[-2])


def extrema_1d(input_, minimum_extrema_distance):
    extrema_primary = torch.zeros_like(input_)
    dx = input_[:, :, 1:] - input_[:, :, :-1]
    dx_padright_greater = functional.pad(dx, [0, 1]) > 0
    dx_padleft_less = functional.pad(dx, [1, 0]) <= 0
    sign = (1 - torch.sign(input_)).bool()
    valleys = dx_padright_greater & dx_padleft_less & sign
    peaks = ~dx_padright_greater & ~dx_padleft_less & ~sign
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for (index, (x_, e_)) in enumerate(zip(input_, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices][:, 0]
        extrema_is_secondary = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for (index_, extrema_index) in enumerate(extrema_indices_sorted):
            if not extrema_is_secondary[index_]:
                extrema_indices_r = extrema_indices_sorted >= extrema_index - minimum_extrema_distance
                extrema_indices_l = extrema_indices_sorted <= extrema_index + minimum_extrema_distance
                extrema_indices_m = extrema_indices_r & extrema_indices_l
                extrema_is_secondary = extrema_is_secondary | extrema_indices_m
                extrema_is_secondary[index_] = False
        extrema_primary_indices = extrema_indices_sorted[~extrema_is_secondary]
        extrema_primary[index, :, extrema_primary_indices] = x_[0, extrema_primary_indices]
    return extrema_primary


def extrema_2d(input_, minimum_extrema_distance):
    extrema_primary = torch.zeros_like(input_)
    dx = input_[:, :, :, 1:] - input_[:, :, :, :-1]
    dy = input_[:, :, 1:, :] - input_[:, :, :-1, :]
    dx_padright_greater = functional.pad(dx, [0, 1, 0, 0]) > 0
    dx_padleft_less = functional.pad(dx, [1, 0, 0, 0]) <= 0
    dy_padright_greater = functional.pad(dy, [0, 0, 0, 1]) > 0
    dy_padleft_less = functional.pad(dy, [0, 0, 1, 0]) <= 0
    sign = (1 - torch.sign(input_)).bool()
    valleys_x = dx_padright_greater & dx_padleft_less & sign
    valleys_y = dy_padright_greater & dy_padleft_less & sign
    peaks_x = ~dx_padright_greater & ~dx_padleft_less & ~sign
    peaks_y = ~dy_padright_greater & ~dy_padleft_less & ~sign
    peaks = peaks_x & peaks_y
    valleys = valleys_x & valleys_y
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for (index, (x_, e_)) in enumerate(zip(input_, extrema)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices]
        extrema_is_secondary = torch.zeros_like(extrema_indices_indices, dtype=torch.bool)
        for (index_, (extrema_index_x, extrema_index_y)) in enumerate(extrema_indices_sorted):
            if not extrema_is_secondary[index_]:
                extrema_indices_r = extrema_indices_sorted[:, 0] >= extrema_index_x - minimum_extrema_distance[0]
                extrema_indices_l = extrema_indices_sorted[:, 0] <= extrema_index_x + minimum_extrema_distance[0]
                extrema_indices_t = extrema_indices_sorted[:, 1] >= extrema_index_y - minimum_extrema_distance[1]
                extrema_indices_b = extrema_indices_sorted[:, 1] <= extrema_index_y + minimum_extrema_distance[1]
                extrema_indices_m = extrema_indices_r & extrema_indices_l & extrema_indices_t & extrema_indices_b
                extrema_is_secondary = extrema_is_secondary | extrema_indices_m
                extrema_is_secondary[index_] = False
        extrema_primary_indices = extrema_indices_sorted[~extrema_is_secondary]
        for extrema_primary_index in extrema_primary_indices:
            extrema_primary[index, :, extrema_primary_index[0], extrema_primary_index[1]] = x_[0, extrema_primary_index[0], extrema_primary_index[1]]
    return extrema_primary


def extrema_pool_indices_1d(input_, kernel_size):
    extrema_primary = torch.zeros_like(input_)
    (_, extrema_indices) = functional.max_pool1d(abs(input_), kernel_size, return_indices=True)
    return extrema_primary.scatter(-1, extrema_indices, input_.gather(-1, extrema_indices))


def extrema_pool_indices_2d(input_, kernel_size):
    x_flattened = input_.view(input_.shape[0], -1)
    extrema_primary = torch.zeros_like(x_flattened)
    (_, extrema_indices) = functional.max_pool2d(abs(input_), kernel_size, return_indices=True)
    return extrema_primary.scatter(-1, extrema_indices[..., 0, 0], x_flattened.gather(-1, extrema_indices[..., 0, 0])).view(input_.shape)


def main():
    artifacts_dir = environ['ARTIFACTS_DIR']
    full = environ['FULL']
    plt.rcParams['font.size'] = 20
    plt.rcParams['image.interpolation'] = 'none'
    plt.rcParams['savefig.bbox'] = 'tight'
    plt.rcParams['savefig.format'] = 'pdf'
    epochs_physionet_num = 30
    epochs_num = 5
    kernel_size_physionet_range = range(1, 250)
    uci_epilepsy_training_range = range(8740)
    uci_epilepsy_validation_range = range(1380)
    uci_epilepsy_test_range = range(1380)
    mnist_fashionmnist_training_range_list = [range(50000), range(50000)]
    mnist_fashionmnist_validation_range_list = [range(50000, 60000), range(50000, 60000)]
    mnist_fashionmnist_test_range_list = [range(10000), range(10000)]
    if not full:
        epochs_physionet_num = 3
        epochs_num = 2
        kernel_size_physionet_range = range(1, 10)
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
    kernel_size_uci_epilepsy_range = range(8, 16)
    kernel_size_mnist_fashionmnist_range_list = [range(1, 7), range(1, 7)]
    dataset_name_list = ['apnea-ecg', 'bidmc', 'bpssrat', 'cebsdb', 'ctu-uhb-ctgdb', 'drivedb', 'emgdb', 'mitdb', 'noneeg', 'prcp', 'shhpsgdb', 'slpdb', 'sufhsdb', 'voiced', 'wrist']
    xlim_weight_list = [74, 113, 10, 71, 45, 20, 9, 229, 37, 105, 15, 232, 40, 70, 173]
    sparse_activation_list = [Identity1D, Relu1D, TopKAbsolutes1D, ExtremaPoolIndices1D, Extrema1D]
    kernel_size_list_list = [[kernel_size_physionet] for kernel_size_physionet in kernel_size_physionet_range]
    batch_size = 2
    lr = 0.01
    sparse_activation_color_list = plt.rcParams['axes.prop_cycle'].by_key()['color']
    results_physionet_row_list_list = []
    flithos_mean_array = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list)))
    flithos_all_validation_array = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list), epochs_physionet_num))
    kernel_size_best_array = np.zeros((len(sparse_activation_list), len(dataset_name_list)), dtype=int)
    gaussian_kde_input_array = np.zeros((len(sparse_activation_list), len(dataset_name_list), len(kernel_size_list_list), 2))
    for (dataset_name_index, (dataset_name, xlim_weight)) in enumerate(zip(dataset_name_list, xlim_weight_list)):
        results_physionet_row_list = []
        dataset_training = PhysionetDataset(artifacts_dir, dataset_name, 'training')
        dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, shuffle=True)
        dataset_validation = PhysionetDataset(artifacts_dir, dataset_name, 'validation')
        dataloader_validation = DataLoader(dataset=dataset_validation)
        dataset_test = PhysionetDataset(artifacts_dir, dataset_name, 'test')
        dataloader_test = DataLoader(dataset=dataset_test)
        (fig, ax_main) = plt.subplots(constrained_layout=True, figsize=(6, 6))
        for (sparse_activation_index, (sparse_activation, sparse_activation_color, sparse_activation_name)) in enumerate(zip(sparse_activation_list, sparse_activation_color_list, sparse_activation_name_list)):
            flithos_mean_best = float('inf')
            for (kernel_size_list_index, kernel_size_list) in enumerate(kernel_size_list_list):
                flithos_epoch_mean_best = float('inf')
                if sparse_activation == TopKAbsolutes1D:
                    sparsity_density_list = [int(dataset_test.data.shape[-1] / kernel_size) for kernel_size in kernel_size_list]
                elif sparse_activation == Extrema1D:
                    sparsity_density_list = np.clip([kernel_size - 3 for kernel_size in kernel_size_list], 1, 999).tolist()
                else:
                    sparsity_density_list = kernel_size_list
                sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
                model = SAN1d(kernel_size_list, sparse_activation_list_).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
                for epoch in range(epochs_physionet_num):
                    train_model_unsupervised(model, optimizer, dataloader_training)
                    (flithos_epoch, *_) = validate_or_test_model_unsupervised(dataloader_validation, hook_handle_list, model)
                    flithos_all_validation_array[sparse_activation_index, dataset_name_index, kernel_size_list_index, epoch] = flithos_epoch.mean()
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        model_epoch_best = model
                        flithos_epoch_mean_best = flithos_epoch.mean()
                (flithos_epoch_best, inverse_compression_ratio_epoch_best, reconstruction_loss_epoch_best) = validate_or_test_model_unsupervised(dataloader_test, hook_handle_list, model_epoch_best)
                flithos_mean_array[sparse_activation_index, dataset_name_index, kernel_size_list_index] = flithos_epoch_best.mean()
                plt.sca(ax_main)
                plt.plot(reconstruction_loss_epoch_best.mean(), inverse_compression_ratio_epoch_best.mean(), 'o', c=sparse_activation_color, markersize=3)
                gaussian_kde_input_array[sparse_activation_index, dataset_name_index, kernel_size_list_index] = [reconstruction_loss_epoch_best.mean(), inverse_compression_ratio_epoch_best.mean()]
                if flithos_mean_array[sparse_activation_index, dataset_name_index, kernel_size_list_index] < flithos_mean_best:
                    kernel_size_best_array[sparse_activation_index, dataset_name_index] = kernel_size_list[0]
                    inverse_compression_ratio_best = inverse_compression_ratio_epoch_best
                    reconstruction_loss_best = reconstruction_loss_epoch_best
                    flithos_mean_best = flithos_mean_array[sparse_activation_index, dataset_name_index, kernel_size_list_index]
                    model_best = model_epoch_best
            results_physionet_row_list.extend([kernel_size_best_array[sparse_activation_index, dataset_name_index], inverse_compression_ratio_best.mean(), reconstruction_loss_best.mean(), flithos_mean_best])
            save_images_1d(artifacts_dir, dataset_test[0][0][0], dataset_name, model_best, sparse_activation_name.lower().replace(' ', '-'), xlim_weight)
            ax_main.arrow(reconstruction_loss_best.mean(), inverse_compression_ratio_best.mean(), 1.83 - reconstruction_loss_best.mean(), 2.25 - 0.5 * sparse_activation_index - inverse_compression_ratio_best.mean())
            fig.add_axes([0.75, 0.81 - 0.165 * sparse_activation_index, 0.1, 0.1], facecolor='y')
            plt.plot(model_best.weights_list[0].flip(0).cpu().detach().numpy().T, c=sparse_activation_color)
            plt.xlim([0, xlim_weight])
            plt.xticks([])
            plt.yticks([])
        results_physionet_row_list_list.append(results_physionet_row_list)
        plt.sca(ax_main)
        plt.xlim([0, 2.5])
        plt.ylim([0, 2.5])
        plt.xlabel('$\\tilde{\\mathcal{L}}$')
        plt.ylabel('$CR^{-1}$')
        plt.grid(True)
        plt.title(dataset_name)
        plt.axhspan(2, 2.5, alpha=0.3, color='r')
        plt.axhspan(1, 2, alpha=0.3, color='orange')
        plt.axvspan(1, 2.5, alpha=0.3, color='gray')
        wedge = Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
        ax_main.add_patch(wedge)
        plt.savefig(join(artifacts_dir, dataset_name))
        plt.close()
    header = ['$m$', '$CR^{-1}$', '$\\tilde{\\mathcal{L}}$', '$\\bar\\varphi$']
    columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    df = pd.DataFrame(results_physionet_row_list_list, columns=columns, index=dataset_name_list)
    df.index.names = ['Datasets']
    styler = df.style
    styler.format(precision=2, formatter={columns[0]: '{:.0f}', columns[4]: '{:.0f}', columns[8]: '{:.0f}', columns[12]: '{:.0f}', columns[16]: '{:.0f}'})
    styler.to_latex(join(artifacts_dir, 'table-flithos-variable-kernel-size.tex'), hrules=True, multicol_align='c')
    (fig, ax) = plt.subplots(constrained_layout=True, figsize=(6, 6))
    var = np.zeros((len(dataset_name_list), epochs_physionet_num))
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for (index, (sparse_activation, sparse_activation_name, sparse_activation_color, kernel_size_best, flithos_all_validation_element_array)) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, kernel_size_best_array, flithos_all_validation_array)):
        t_range = range(1, flithos_all_validation_element_array.shape[-1] + 1)
        for (index_, (c_, k_)) in enumerate(zip(flithos_all_validation_element_array, kernel_size_best)):
            var[index_] = c_[k_ - 1]
        mu = var.mean(axis=0)
        sigma = var.std(axis=0)
        ax.fill_between(t_range, mu + sigma, mu - sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t_range, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, fontsize=12, loc='lower left')
    plt.xlabel('epochs')
    plt.ylabel('$\\bar\\varphi$')
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 2.5])
    plt.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(join(artifacts_dir, 'mean-flithos-validation-epochs'))
    plt.close()
    (fig, ax) = plt.subplots(constrained_layout=True, figsize=(6, 6))
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for (index, (sparse_activation, sparse_activation_name, sparse_activation_color, flithos_mean_element_array)) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, flithos_mean_array)):
        t_range = range(1, flithos_mean_element_array.shape[1] + 1)
        mu = flithos_mean_element_array.mean(axis=0)
        sigma = flithos_mean_element_array.std(axis=0)
        ax.fill_between(t_range, mu + sigma, mu - sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t_range, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, fontsize=12, loc='lower right')
    plt.xlabel('$m$')
    plt.ylabel('$\\bar\\varphi$')
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
    non_sparse_model_description_patch = Patch(color='r', alpha=0.3, label='non-sparse model description')
    worse_cr_than_original_data_patch = Patch(color='orange', alpha=0.3, label='worse $CR^{-1}$ than original data')
    worse_l_than_constant_prediction_patch = Patch(color='gray', alpha=0.3, label='worse $\\tilde{\\mathcal{L}}$ than constant prediction')
    varphi_less_than_one_patch = Patch(color='g', alpha=0.3, label='$\\bar\\varphi < 1$')
    line2d_list = []
    for (sparse_activation_name, sparse_activation_color) in zip(sparse_activation_name_list, sparse_activation_color_list):
        line2d_list.append(Line2D([0], [0], marker='o', color='w', label=sparse_activation_name, markerfacecolor=sparse_activation_color))
    legend_handle_list = [non_sparse_model_description_patch, worse_cr_than_original_data_patch, worse_l_than_constant_prediction_patch, varphi_less_than_one_patch, line2d_list[0], line2d_list[1], line2d_list[2], line2d_list[3], line2d_list[4]]
    fig_legend.legend(handles=legend_handle_list, fontsize=22, loc='upper center')
    plt.savefig(join(artifacts_dir, 'legend'))
    plt.close()
    (fig, ax) = plt.subplots(constrained_layout=True, figsize=(6, 6))
    gaussian_kde_input_array = gaussian_kde_input_array.reshape(gaussian_kde_input_array.shape[0], -1, 2)
    nbins = 200
    (yi, xi) = np.mgrid[0:2.5:nbins * 1j, 0:2.5:nbins * 1j]
    for (sparse_activation_color, gaussian_kde_input_element_array) in zip(sparse_activation_color_list, gaussian_kde_input_array):
        gkde = gaussian_kde(gaussian_kde_input_element_array.T)
        zi = gkde(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
        plt.contourf(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
    plt.axhspan(2, 2.5, alpha=0.3, color='r')
    plt.axhspan(1, 2, alpha=0.3, color='orange')
    plt.axvspan(1, 2.5, alpha=0.3, color='gray')
    wedge = Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
    ax.add_patch(wedge)
    plt.xlabel('$\\tilde{\\mathcal{L}}$')
    plt.ylabel('$CR^{-1}$')
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.grid(True)
    plt.savefig(join(artifacts_dir, 'crrl-density-plot'))
    plt.close()
    batch_size = 64
    lr = 0.01
    uci_epilepsy_dir = join(artifacts_dir, 'UCI-epilepsy')
    dataset_training = UCIepilepsyDataset(uci_epilepsy_dir, 'training')
    dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    dataset_validation = UCIepilepsyDataset(uci_epilepsy_dir, 'validation')
    dataloader_validation = DataLoader(dataset=dataset_validation, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    dataset_test = UCIepilepsyDataset(uci_epilepsy_dir, 'test')
    dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_test_range))
    accuracy_best = 0
    model_supervised = CNN(len(dataset_training.classes.unique())).to(device)
    optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
    for epoch in range(epochs_num):
        model_supervised.train()
        for (data, target) in dataloader_training:
            data = data.to(device)
            target = target.to(device)
            output = model_supervised(data)
            classification_loss = functional.cross_entropy(output, target)
            optimizer.zero_grad()
            classification_loss.backward()
            optimizer.step()
        predictions_correct_num = 0
        predictions_num = 0
        model_supervised.eval()
        with torch.no_grad():
            for (data, target) in dataloader_validation:
                data = data.to(device)
                target = target.to(device)
                output = model_supervised(data)
                prediction = output.argmax(dim=1)
                predictions_correct_num += sum(prediction == target).item()
                predictions_num += output.shape[0]
        accuracy = 100 * predictions_correct_num / predictions_num
        if accuracy_best < accuracy:
            model_supervised_best = model_supervised
            accuracy_best = accuracy
    model_supervised.eval()
    with torch.no_grad():
        for (data, target) in dataloader_test:
            data = data.to(device)
            target = target.to(device)
            output = model_supervised_best(data)
            prediction = output.argmax(dim=1)
            predictions_correct_num += sum(prediction == target).item()
            predictions_num += output.shape[0]
    accuracy_uci_epilepsy = 100 * predictions_correct_num / predictions_num
    dataset_name = 'UCI-epilepsy'
    sparse_activation_list = [Identity1D, Relu1D, TopKAbsolutes1D, ExtremaPoolIndices1D, Extrema1D]
    kernel_size_list_list = [2 * [kernel_size_uci_epilepsy] for kernel_size_uci_epilepsy in kernel_size_uci_epilepsy_range]
    batch_size = 64
    lr = 0.01
    results_supervised_row_list_list = []
    dataset_training = UCIepilepsyDataset(uci_epilepsy_dir, 'training')
    dataloader_training = DataLoader(dataset=dataset_training, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    dataset_validation = UCIepilepsyDataset(uci_epilepsy_dir, 'validation')
    dataloader_validation = DataLoader(dataset=dataset_validation, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    dataset_test = UCIepilepsyDataset(uci_epilepsy_dir, 'test')
    dataloader_test = DataLoader(dataset=dataset_test, sampler=SubsetRandomSampler(uci_epilepsy_test_range))
    for (kernel_size_list_index, kernel_size_list) in enumerate(kernel_size_list_list):
        results_supervised_row_list = []
        for (sparse_activation_index, (sparse_activation, sparse_activation_name)) in enumerate(zip(sparse_activation_list, sparse_activation_name_list)):
            if sparse_activation == TopKAbsolutes1D:
                sparsity_density_list = [int(dataset_test.data.shape[-1] / kernel_size) for kernel_size in kernel_size_list]
            elif sparse_activation == Extrema1D:
                sparsity_density_list = np.clip([kernel_size - 2 for kernel_size in kernel_size_list], 1, 999).tolist()
            else:
                sparsity_density_list = kernel_size_list
            sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
            model = SAN1d(kernel_size_list, sparse_activation_list_).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
            flithos_epoch_mean_best = float('inf')
            for epoch in range(epochs_num):
                train_model_unsupervised(model, optimizer, dataloader_training)
                (flithos_epoch, *_) = validate_or_test_model_unsupervised(dataloader_validation, hook_handle_list, model)
                if flithos_epoch.mean() < flithos_epoch_mean_best:
                    model_epoch_best = model
                    flithos_epoch_mean_best = flithos_epoch.mean()
            for weights in model.weights_list:
                weights.requires_grad_(False)
            flithos_epoch_mean_best = float('inf')
            model_supervised = CNN(len(dataset_training.classes.unique())).to(device).to(device)
            optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
            for epoch in range(epochs_num):
                train_model_supervised(model_supervised, model_epoch_best, optimizer, dataloader_training)
                (flithos_epoch, *_) = validate_or_test_model_unsupervised(dataloader_validation, hook_handle_list, model_epoch_best)
                if flithos_epoch.mean() < flithos_epoch_mean_best:
                    model_supervised_best = model_supervised
                    model_best = model_epoch_best
                    flithos_epoch_mean_best = flithos_epoch.mean()
            (flithos, inverse_compression_ratio, reconstruction_loss, accuracy) = validate_or_test_model_supervised(dataloader_test, hook_handle_list, model_supervised_best, model_best)
            results_supervised_row_list.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - accuracy_uci_epilepsy])
            if kernel_size_list[0] == 10:
                save_images_1d(artifacts_dir, dataset_test[0][0][0], dataset_name, model_best, sparse_activation_name.lower().replace(' ', '-'), kernel_size_list[0])
        results_supervised_row_list_list.append(results_supervised_row_list)
    header = ['$CR^{-1}$', '$\\tilde{\\mathcal{L}}$', '$\\bar\\varphi$', 'A\\textsubscript{$\\pm$\\%}']
    columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    df = pd.DataFrame(results_supervised_row_list_list, columns=columns, index=kernel_size_uci_epilepsy_range)
    df.index.names = ['$m$']
    styler = df.style
    styler.format(precision=2, formatter={columns[3]: '{:.1f}', columns[7]: '{:.1f}', columns[11]: '{:.1f}', columns[15]: '{:.1f}', columns[19]: '{:.1f}'})
    styler.to_latex(join(artifacts_dir, 'table-uci-epilepsy-supervised.tex'), hrules=True, multicol_align='c')
    dataset_name_list = ['MNIST', 'FashionMNIST']
    dataset_list = [MNIST, FashionMNIST]
    accuracy_mnist_fashionmnist_supervised_list = [0, 0]
    for (dataset_name_index, (dataset_name, dataset, kernel_size_mnist_fashionmnist_range, mnist_fashionmnist_training_range, mnist_fashionmnist_validation_range, mnist_fashionmnist_test_range)) in enumerate(zip(dataset_name_list, dataset_list, kernel_size_mnist_fashionmnist_range_list, mnist_fashionmnist_training_range_list, mnist_fashionmnist_validation_range_list, mnist_fashionmnist_test_range_list)):
        batch_size = 64
        lr = 0.01
        dataset_training_validation = dataset(artifacts_dir, download=True, train=True, transform=ToTensor())
        dataloader_training = DataLoader(dataset_training_validation, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_fashionmnist_training_range))
        dataloader_validation = DataLoader(dataset_training_validation, sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range), batch_size=batch_size)
        dataset_test = dataset(artifacts_dir, train=False, transform=ToTensor())
        dataloader_test = DataLoader(dataset_test, sampler=SubsetRandomSampler(mnist_fashionmnist_test_range))
        accuracy_best = 0
        model_supervised = FNN(len(dataset_training_validation.classes), dataset_training_validation.data[0]).to(device)
        optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
        for epoch in range(epochs_num):
            model_supervised.train()
            for (data, target) in dataloader_training:
                data = data.to(device)
                target = target.to(device)
                output = model_supervised(data)
                classification_loss = functional.cross_entropy(output, target)
                optimizer.zero_grad()
                classification_loss.backward()
                optimizer.step()
            predictions_correct_num = 0
            predictions_num = 0
            model_supervised.eval()
            with torch.no_grad():
                for (data, target) in dataloader_validation:
                    data = data.to(device)
                    target = target.to(device)
                    output = model_supervised(data)
                    prediction = output.argmax(dim=1)
                    predictions_correct_num += sum(prediction == target).item()
                    predictions_num += output.shape[0]
            accuracy = 100 * predictions_correct_num / predictions_num
            if accuracy_best < accuracy:
                model_supervised_best = model_supervised
                accuracy_best = accuracy
        model_supervised.eval()
        with torch.no_grad():
            for (data, target) in dataloader_test:
                data = data.to(device)
                target = target.to(device)
                output = model_supervised_best(data)
                prediction = output.argmax(dim=1)
                predictions_correct_num += sum(prediction == target).item()
                predictions_num += output.shape[0]
        accuracy_mnist_fashionmnist_supervised_list[dataset_name_index] = 100 * predictions_correct_num / predictions_num
        sparse_activation_list = [Identity2D, Relu2D, TopKAbsolutes2D, ExtremaPoolIndices2D, Extrema2D]
        kernel_size_list_list = [2 * [kernel_size_mnist_fashionmnist] for kernel_size_mnist_fashionmnist in kernel_size_mnist_fashionmnist_range]
        batch_size = 64
        lr = 0.01
        results_supervised_row_list_list = []
        dataset_training_validation = dataset(artifacts_dir, download=True, train=True, transform=ToTensor())
        dataloader_training = DataLoader(dataset_training_validation, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_fashionmnist_training_range))
        dataloader_validation = DataLoader(dataset_training_validation, sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range))
        dataset_test = dataset(artifacts_dir, train=False, transform=ToTensor())
        dataloader_test = DataLoader(dataset_test, sampler=SubsetRandomSampler(mnist_fashionmnist_test_range))
        for (kernel_size_list_index, kernel_size_list) in enumerate(kernel_size_list_list):
            results_supervised_row_list = []
            for (sparse_activation_index, (sparse_activation, sparse_activation_name)) in enumerate(zip(sparse_activation_list, sparse_activation_name_list)):
                if sparse_activation == TopKAbsolutes2D:
                    sparsity_density_list = [int(dataset_test.data.shape[-1] / kernel_size) ** 2 for kernel_size in kernel_size_list]
                elif sparse_activation == Extrema2D:
                    sparsity_density_list = np.clip([kernel_size - 2 for kernel_size in kernel_size_list], 1, 999).tolist()
                    sparsity_density_list = [[sparsity_density, sparsity_density] for sparsity_density in sparsity_density_list]
                else:
                    sparsity_density_list = kernel_size_list
                sparse_activation_list_ = [sparse_activation(sparsity_density) for sparsity_density in sparsity_density_list]
                model = SAN2d(kernel_size_list, sparse_activation_list_).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
                flithos_epoch_mean_best = float('inf')
                for epoch in range(epochs_num):
                    train_model_unsupervised(model, optimizer, dataloader_training)
                    (flithos_epoch, *_) = validate_or_test_model_unsupervised(dataloader_validation, hook_handle_list, model)
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        model_epoch_best = model
                        flithos_epoch_mean_best = flithos_epoch.mean()
                for weights in model.weights_list:
                    weights.requires_grad_(False)
                flithos_epoch_mean_best = float('inf')
                model_supervised = FNN(len(dataset_training_validation.classes), dataset_training_validation.data[0]).to(device)
                optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
                for epoch in range(epochs_num):
                    train_model_supervised(model_supervised, model_epoch_best, optimizer, dataloader_training)
                    (flithos_epoch, *_) = validate_or_test_model_unsupervised(dataloader_validation, hook_handle_list, model_epoch_best)
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        model_supervised_best = model_supervised
                        model_best = model_epoch_best
                        flithos_epoch_mean_best = flithos_epoch.mean()
                (flithos, inverse_compression_ratio, reconstruction_loss, accuracy) = validate_or_test_model_supervised(dataloader_test, hook_handle_list, model_supervised_best, model_best)
                results_supervised_row_list.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - accuracy_mnist_fashionmnist_supervised_list[dataset_name_index]])
                if kernel_size_list[0] == 4:
                    save_images_2d(artifacts_dir, dataset_test[0][0][0], dataset_name, model_best, sparse_activation_name.lower().replace(' ', '-'))
            results_supervised_row_list_list.append(results_supervised_row_list)
        header = ['$CR^{-1}$', '$\\tilde{\\mathcal{L}}$', '$\\bar\\varphi$', 'A\\textsubscript{$\\pm$\\%}']
        columns = pd.MultiIndex.from_product([sparse_activation_name_list, header])
        df = pd.DataFrame(results_supervised_row_list_list, columns=columns, index=kernel_size_mnist_fashionmnist_range)
        df.index.names = ['$m$']
        styler = df.style
        styler.format(precision=2, formatter={columns[3]: '{:.1f}', columns[7]: '{:.1f}', columns[11]: '{:.1f}', columns[15]: '{:.1f}', columns[19]: '{:.1f}'})
        styler.to_latex(join(artifacts_dir, f'table-{dataset_name.lower()}-supervised.tex'), hrules=True, multicol_align='c')
    df = pd.DataFrame({'key': ['uci-epilepsy-supervised-accuracy', 'mnist-supervised-accuracy', 'fashionmnist-supervised-accuracy'], 'value': [accuracy_uci_epilepsy, accuracy_mnist_fashionmnist_supervised_list[0], accuracy_mnist_fashionmnist_supervised_list[1]]})
    df.to_csv(join(artifacts_dir, 'keys-values.csv'), index=False, float_format='%.2f')


def save_images_1d(artifacts_dir, data, dataset_name, model, sparse_activation_name, xlim_weight):
    model = model.to('cpu')
    (_, ax) = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(True)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-signal'))
    plt.close()
    hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
    model.eval()
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_list = []
        for hook_handle in hook_handle_list:
            activations_list.append(hook_handle.output)
        activations_list = torch.stack(activations_list, 1)
        for (weights_index, (weights, activations)) in enumerate(zip(model.weights_list, activations_list[0, :, 0])):
            (_, ax) = plt.subplots(figsize=(2, 2.2))
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.xaxis.get_offset_text().set_visible(False)
            ax.yaxis.get_offset_text().set_visible(False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(weights.cpu().detach().numpy(), 'r')
            plt.xlim([0, xlim_weight])
            if dataset_name == 'apnea-ecg':
                plt.ylabel(sparse_activation_name, fontsize=20)
            if sparse_activation_name == 'relu':
                plt.title(dataset_name, fontsize=20)
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-kernel-{weights_index}'))
            plt.close()
            similarity = functional.conv1d(data.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            (_, ax) = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            plt.plot(similarity.cpu().detach().numpy(), 'g')
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-similarity-{weights_index}'))
            plt.close()
            (_, ax) = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            peaks = torch.nonzero(activations, as_tuple=False)[:, 0]
            plt.plot(similarity.cpu().detach().numpy(), 'g', alpha=0.5)
            if peaks.shape[0] != 0:
                plt.stem(peaks.cpu().detach().numpy(), activations[peaks.cpu().detach().numpy()].cpu().detach().numpy(), linefmt='c', basefmt=' ')
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-activations-{weights_index}'))
            plt.close()
            reconstruction = functional.conv1d(activations.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            (_, ax) = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(True)
            plt.autoscale(enable=True, axis='x', tight=True)
            reconstruction = reconstruction.cpu().detach().numpy()
            lefts = peaks - weights.shape[0] / 2
            rights = peaks + weights.shape[0] / 2
            if weights.shape[0] % 2 == 1:
                rights += 1
            step = np.zeros_like(reconstruction, dtype=bool)
            lefts[lefts < 0] = 0
            rights[rights > reconstruction.shape[0]] = reconstruction.shape[0]
            for (left, right) in zip(lefts, rights):
                step[int(left):int(right)] = True
            pos_signal = reconstruction.copy()
            neg_signal = reconstruction.copy()
            pos_signal[step] = np.nan
            neg_signal[~step] = np.nan
            plt.plot(pos_signal)
            plt.plot(neg_signal, color='r')
            plt.ylim([data.min(), data.max()])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-reconstruction-{weights_index}'))
            plt.close()
        (_, ax) = plt.subplots()
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True)
        plt.autoscale(enable=True, axis='x', tight=True)
        plt.plot(data.cpu().detach().numpy(), alpha=0.5)
        plt.plot(reconstructed[0, 0].cpu().detach().numpy(), 'r')
        plt.ylim([data.min(), data.max()])
        plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_list)}-reconstructed'))
        plt.close()


def save_images_2d(artifacts_dir, data, dataset_name, model, sparse_activation_name):
    model = model.to('cpu')
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data.cpu().detach().numpy(), cmap='twilight', vmin=-2, vmax=2)
    plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-signal'))
    plt.close()
    hook_handle_list = [Hook(sparse_activation_) for sparse_activation_ in model.sparse_activation_list]
    model.eval()
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_list = []
        for hook_handle in hook_handle_list:
            activations_list.append(hook_handle.output)
        activations_list = torch.stack(activations_list, 1)
        for (weights_index, (weights, activations)) in enumerate(zip(model.weights_list, activations_list[0, :, 0])):
            plt.figure(figsize=(4.8 / 2, 4.8 / 2))
            plt.imshow(weights.flip(0).flip(1).cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(weights).max(), vmax=2 * abs(weights).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-kernel-{weights_index}'))
            plt.close()
            similarity = functional.conv2d(data.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(similarity.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(similarity).max(), vmax=2 * abs(similarity).max())
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-similarity-{weights_index}'))
            plt.close()
            plt.figure()
            plt.imshow(activations.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(activations).max(), vmax=2 * abs(activations).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-activations-{weights_index}'))
            plt.close()
            reconstruction = functional.conv2d(activations.unsqueeze(0).unsqueeze(0), weights.unsqueeze(0).unsqueeze(0), padding='same')[0, 0]
            plt.figure()
            plt.imshow(reconstruction.cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(reconstruction).max(), vmax=2 * abs(reconstruction).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-reconstruction-{weights_index}'))
            plt.close()
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='twilight', vmin=-2 * abs(reconstructed).max(), vmax=2 * abs(reconstructed).max())
        plt.savefig(join(artifacts_dir, f'{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_list)}-reconstructed'))
        plt.close()


def topk_absolutes_1d(input_, topk):
    extrema_primary = torch.zeros_like(input_)
    (_, extrema_indices) = torch.topk(abs(input_), topk)
    return extrema_primary.scatter(-1, extrema_indices, input_.gather(-1, extrema_indices))


def topk_absolutes_2d(input_, topk):
    x_flattened = input_.view(input_.shape[0], -1)
    extrema_primary = torch.zeros_like(x_flattened)
    (_, extrema_indices) = torch.topk(abs(x_flattened), topk)
    return extrema_primary.scatter(-1, extrema_indices, x_flattened.gather(-1, extrema_indices)).view(input_.shape)


def train_model_supervised(model_supervised, model_unsupervised, optimizer, dataloader_training):
    device = next(model_supervised.parameters()).device
    model_supervised.train()
    for (data, target) in dataloader_training:
        data = data.to(device)
        target = target.to(device)
        data_reconstructed = model_unsupervised(data)
        output = model_supervised(data_reconstructed)
        classification_loss = functional.cross_entropy(output, target)
        optimizer.zero_grad()
        classification_loss.backward()
        optimizer.step()


def train_model_unsupervised(model, optimizer, dataloader_training):
    device = next(model.parameters()).device
    model.train()
    for (data, _) in dataloader_training:
        data = data.to(device)
        data_reconstructed = model(data)
        reconstruction_loss = functional.l1_loss(data, data_reconstructed)
        optimizer.zero_grad()
        reconstruction_loss.backward()
        optimizer.step()


def validate_or_test_model_supervised(dataloader, hook_handle_list, model_supervised, model_unsupervised):
    device = next(model_supervised.parameters()).device
    predictions_correct_num = 0
    activations_num = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    model_supervised.eval()
    with torch.no_grad():
        for (index, (data, target)) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            data_reconstructed = model_unsupervised(data)
            activations_list = []
            for hook_handle in hook_handle_list:
                activations_list.append(hook_handle.output)
            activations_list = torch.stack(activations_list, 1)
            reconstruction_loss[index] = functional.l1_loss(data, data_reconstructed) / functional.l1_loss(data, torch.zeros_like(data))
            activations_num[index] = torch.nonzero(activations_list, as_tuple=False).shape[0]
            output = model_supervised(data_reconstructed)
            prediction = output.argmax(dim=1)
            predictions_correct_num += sum(prediction == target).item()
    inverse_compression_ratio = calculate_inverse_compression_ratio(activations_num, data, model_unsupervised)
    flithos = np.mean([np.sqrt(icr ** 2 + rl ** 2) for (icr, rl) in zip(inverse_compression_ratio, reconstruction_loss)])
    return (flithos, inverse_compression_ratio, reconstruction_loss, 100 * predictions_correct_num / len(dataloader.sampler))


def validate_or_test_model_unsupervised(dataloader, hook_handle_list, model):
    device = next(model.parameters()).device
    activations_num = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    model.eval()
    with torch.no_grad():
        for (index, (data, _)) in enumerate(dataloader):
            data = data.to(device)
            data_reconstructed = model(data)
            activations_list = []
            for hook_handle in hook_handle_list:
                activations_list.append(hook_handle.output)
            activations_list = torch.stack(activations_list, 1)
            reconstruction_loss[index] = functional.l1_loss(data, data_reconstructed) / functional.l1_loss(data, torch.zeros_like(data))
            activations_num[index] = torch.nonzero(activations_list, as_tuple=False).shape[0]
    inverse_compression_ratio = calculate_inverse_compression_ratio(activations_num, data, model)
    flithos = np.mean([np.sqrt(icr ** 2 + rl ** 2) for (icr, rl) in zip(inverse_compression_ratio, reconstruction_loss)])
    return (flithos, inverse_compression_ratio, reconstruction_loss)


if __name__ == '__main__':
    main()