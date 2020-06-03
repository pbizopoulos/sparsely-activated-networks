#!/usr/bin/env python
import argparse
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import torch

from matplotlib import patches
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

from san_1d import SAN1d, identity_1d, relu_1d, topk_absolutes_1d, extrema_pool_indices_1d, extrema_1d
from san_2d import SAN2d, identity_2d, relu_2d, topk_absolutes_2d, extrema_pool_indices_2d, extrema_2d
from utilities import calculate_inverse_compression_ratio, FNN, CNN, path_tmp, path_dataset, path_paper
from utilities_1d import save_images_1d, download_physionet, download_uci_epilepsy, PhysionetDataset, UCIepilepsyDataset
from utilities_2d import save_images_2d


def train_unsupervised_model(model, optimizer, training_dataloader, device):
    model.train()
    for data, _ in training_dataloader:
        data = data.to(device)
        optimizer.zero_grad()
        data_reconstructed, *_ = model(data)
        reconstruction_loss = F.l1_loss(data, data_reconstructed)
        reconstruction_loss.backward()
        optimizer.step()

def validate_or_test_unsupervised_model(model, dataloader, device):
    model.eval()
    num_activations = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    with torch.no_grad():
        for index, (data, _) in enumerate(dataloader):
            data = data.to(device)
            data_reconstructed, _, activations_list, _ = model(data)
            reconstruction_loss[index] = F.l1_loss(data, data_reconstructed) / F.l1_loss(data, torch.zeros_like(data))
            num_activations[index] = activations_list.nonzero().shape[0]
    inverse_compression_ratio = calculate_inverse_compression_ratio(model, data, num_activations)
    flithos = np.mean([np.sqrt(i**2 + r**2) for i, r in zip(inverse_compression_ratio, reconstruction_loss)])
    return flithos, inverse_compression_ratio, reconstruction_loss

def train_supervised_model(supervised_model, unsupervised_model, optimizer, training_dataloader, device):
    supervised_model.train()
    for data, target in training_dataloader:
        data = data.to(device)
        target = target.to(device)
        optimizer.zero_grad()
        data_reconstructed, *_ = unsupervised_model(data)
        output = supervised_model(data_reconstructed)
        classification_loss = F.cross_entropy(output, target)
        classification_loss.backward()
        optimizer.step()

def validate_or_test_supervised_model(supervised_model, unsupervised_model, dataloader, device):
    supervised_model.eval()
    correct = 0
    num_activations = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    with torch.no_grad():
        for index, (data, target) in enumerate(dataloader):
            data = data.to(device)
            target = target.to(device)
            data_reconstructed, _, activations_list, _ = unsupervised_model(data)
            reconstruction_loss[index] = F.l1_loss(data, data_reconstructed) / F.l1_loss(data, torch.zeros_like(data))
            num_activations[index] = activations_list.nonzero().shape[0]
            output = supervised_model(data_reconstructed)
            pred = output.argmax(dim=1)
            correct += (pred == target).sum().item()
    inverse_compression_ratio = calculate_inverse_compression_ratio(unsupervised_model, data, num_activations)
    flithos = np.mean([np.sqrt(i**2 + r**2) for i, r in zip(inverse_compression_ratio, reconstruction_loss)])
    return flithos, inverse_compression_ratio, reconstruction_loss, 100 * correct / len(dataloader.sampler)


if __name__ == '__main__':
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    path_images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list = f'{path_paper}/images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list'
    if not os.path.exists(path_images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list):
        os.mkdir(path_images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list)
    if not os.path.exists(path_tmp):
        os.mkdir(path_tmp)
    path_tables = f'{path_paper}/tables'
    if not os.path.exists(path_tables):
        os.mkdir(path_tables)
    sparse_activation_name_list = ['Identity', 'ReLU', 'top-k absolutes', 'Extrema-Pool idx', 'Extrema']
    uci_epilepsy_supervised_accuracy = 0
    mnist_supervised_accuracy = 0
    fashionmnist_supervised_accuracy = 0
    uci_epilepsy_kernel_size_range = range(8, 16)
    mnist_kernel_size_range = range(1, 7)
    fashionmnist_kernel_size_range = range(1, 7)
    parser = argparse.ArgumentParser()
    parser.add_argument('--full', default=False, action='store_true')
    parser.add_argument('--gpu', default=False, action='store_true')
    args = parser.parse_args()
    if args.full:
        num_epochs_physionet = 30
        num_epochs = 5
        physionet_kernel_size_list_list_range = range(1, 250)
        uci_epilepsy_training_range = range(8740)
        uci_epilepsy_validation_range = range(1380)
        uci_epilepsy_test_range = range(1380)
        mnist_training_range = range(50000)
        mnist_validation_range = range(50000, 60000)
        mnist_test_range = range(10000)
        fashionmnist_training_range = range(50000)
        fashionmnist_validation_range = range(50000, 60000)
        fashionmnist_test_range = range(10000)
    else:
        num_epochs_physionet = 3
        num_epochs = 2
        physionet_kernel_size_list_list_range = range(1, 10)
        uci_epilepsy_training_range = range(10)
        uci_epilepsy_validation_range = range(10)
        uci_epilepsy_test_range = range(10)
        mnist_training_range = range(10)
        mnist_validation_range = range(10, 20)
        mnist_test_range = range(10)
        fashionmnist_training_range = range(10)
        fashionmnist_validation_range = range(10, 20)
        fashionmnist_test_range = range(10)
    if args.gpu:
        device = 'cuda'
    else:
        device = 'cpu'

    print('Physionet, X: mean reconstruction loss, Y: mean inverse compression ratio, Color: sparse activation')
    dataset_name_list = ['apnea-ecg', 'bidmc', 'bpssrat', 'cebsdb', 'ctu-uhb-ctgdb', 'drivedb', 'emgdb', 'mitdb', 'noneeg', 'prcp', 'shhpsgdb', 'slpdb', 'sufhsdb', 'voiced', 'wrist']
    xlim_weights_list = [74, 113, 10, 71, 45, 20, 9, 229, 37, 105, 15, 232, 40, 70, 173]
    download_physionet(dataset_name_list)
    sparse_activation_list = [identity_1d, relu_1d, topk_absolutes_1d, extrema_pool_indices_1d, extrema_1d]
    kernel_size_list_list = [[k] for k in physionet_kernel_size_list_list_range]
    batch_size = 2
    lr = 0.01
    for_colors = np.linspace(0.75, 0.25, len(kernel_size_list_list))
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
        fig = plt.figure(constrained_layout=True, figsize=(6, 6))
        ax_main = plt.gca()
        for index_sparse_activation, (sparse_activation, sparse_activation_color) in enumerate(zip(sparse_activation_list, sparse_activation_color_list)):
            mean_flithos_best = float('inf')
            for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
                mean_flithos_epoch_best = float('inf')
                if sparse_activation == extrema_1d:
                    sparsity_density_list = np.clip([k - 3 for k in kernel_size_list], 1, 999).tolist()
                else:
                    sparsity_density_list = kernel_size_list
                model = SAN1d(sparse_activation, kernel_size_list, sparsity_density_list, device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                for epoch in range(num_epochs_physionet):
                    train_unsupervised_model(model, optimizer, training_dataloader, device)
                    flithos_epoch, *_ = validate_or_test_unsupervised_model(model, validation_dataloader, device)
                    flithos_all_validation[index_sparse_activation, index_dataset_name, index_kernel_size_list, epoch] = flithos_epoch.mean()
                    if flithos_epoch.mean() < mean_flithos_epoch_best:
                        model_epoch_best = model
                        mean_flithos_epoch_best = flithos_epoch.mean()
                flithos_epoch_best, inverse_compression_ratio_epoch_best, reconstruction_loss_epoch_best = validate_or_test_unsupervised_model(model_epoch_best, test_dataloader, device)
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
            save_images_1d(model_best, dataset_name, test_dataset[0][0][0], xlim_weights, device)
            ax_main.arrow(reconstruction_loss_best.mean(), inverse_compression_ratio_best.mean(), 1.83 - reconstruction_loss_best.mean(), 2.25 - 0.5*index_sparse_activation - inverse_compression_ratio_best.mean())
            fig.add_axes([0.75, 0.81 - 0.165*index_sparse_activation, .1, .1], facecolor='y')
            plt.plot(model_best.neuron_list[0].weights.flip(0).cpu().detach().numpy().T, c=sparse_activation_color)
            plt.xlim([0, xlim_weights])
            plt.xticks([])
            plt.yticks([])
        physionet_latex_table.append(physionet_latex_table_row)
        plt.sca(ax_main)
        plt.xlim([0, 2.5])
        plt.ylim([0, 2.5])
        plt.xlabel(r'$\tilde{\mathcal{L}}$', fontsize=20)
        plt.ylabel(r'$CR^{-1}$', fontsize=20)
        plt.xticks(fontsize=20)
        plt.yticks(fontsize=20)
        plt.grid(True)
        plt.title(dataset_name, fontsize=20)
        plt.axhspan(2, 2.5, alpha=0.3, color='r')
        plt.axhspan(1, 2, alpha=0.3, color='orange')
        plt.axvspan(1, 2.5, alpha=0.3, color='gray')
        wedge = patches.Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
        ax_main.add_patch(wedge)
        plt.savefig(f'{path_images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list}/{dataset_name}.pdf')
        plt.close()
    header = ['$m$', r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$']
    index = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    physionet_latex_table = np.array(physionet_latex_table).T.tolist()
    df = pd.DataFrame(physionet_latex_table, index=index)
    df = df.T
    df.index = dataset_name_list
    df.index.names = ['Datasets']
    formatters = 5*[lambda x: f'{x:.0f}', lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:.2f}']
    df.to_latex(f'{path_tables}/mean_inverse_compression_ratio_mean_reconstruction_loss_variable_kernel_size.tex', bold_rows=True, escape=False, column_format='l|rrrr|rrrr|rrrr|rrrr|rrrr', multicolumn_format='c', formatters=formatters)

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    ax = plt.gca()
    var = np.zeros((len(dataset_name_list), num_epochs_physionet))
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for index, (sparse_activation, sparse_activation_name, sparse_activation_color, kernel_size_best, c) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, kernel_size_list_best, flithos_all_validation)):
        t = np.arange(1, c.shape[-1] + 1)
        for j, (x_, y_) in enumerate(zip(c, kernel_size_best)):
            var[j] = x_[int(y_ - 1)]
        mu = var.mean(axis=0)
        sigma = var.std(axis=0)
        ax.fill_between(t, mu+sigma, mu-sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, loc='lower left')
    plt.xlabel(r'epochs', fontsize=20)
    plt.ylabel(r'$\bar\varphi$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 2.5])
    plt.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{path_paper}/images_1d/mean_flithos_validation_epochs.pdf')
    plt.close()

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    ax = plt.gca()
    p1 = [0, 0, 0, 0, 0]
    p2 = [0, 0, 0, 0, 0]
    for index, (sparse_activation, sparse_activation_name, sparse_activation_color, c) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, sparse_activation_color_list, mean_flithos)):
        t = np.arange(1, c.shape[1] + 1)
        mu = c.mean(axis=0)
        sigma = c.std(axis=0)
        ax.fill_between(t, mu+sigma, mu-sigma, facecolor=sparse_activation_color, alpha=0.3)
        p1[index] = ax.plot(t, mu, color=sparse_activation_color)
        p2[index] = ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)
    ax.legend([(p2[0][0], p1[0][0]), (p2[1][0], p1[1][0]), (p2[2][0], p1[2][0]), (p2[3][0], p1[3][0]), (p2[4][0], p1[4][0])], sparse_activation_name_list, loc='lower right')
    plt.xlabel(r'$m$', fontsize=20)
    plt.ylabel(r'$\bar\varphi$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.autoscale(enable=True, axis='x', tight=True)
    plt.ylim([0, 2.5])
    plt.grid(True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig(f'{path_paper}/images_1d/mean_flithos_variable_kernel_size_list.pdf')
    plt.close()

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    fig_legend = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2), range(2), range(2))
    legend_elements = [
            patches.Patch(color='r', alpha=0.3, label='non-sparse model description'),
            patches.Patch(color='orange', alpha=0.3, label=r'worse $CR^{-1}$ than original data'),
            patches.Patch(color='gray', alpha=0.3, label=r'worse $\tilde{\mathcal{L}}$ than constant prediction'),
            patches.Patch(color='g', alpha=0.3, label=r'$\bar\varphi < 1$'),
            Line2D([0], [0], marker='o', color='w', label='Identity', markerfacecolor=sparse_activation_color_list[0]),
            Line2D([0], [0], marker='o', color='w', label='ReLU', markerfacecolor=sparse_activation_color_list[1]),
            Line2D([0], [0], marker='o', color='w', label='top-k absolutes', markerfacecolor=sparse_activation_color_list[2]),
            Line2D([0], [0], marker='o', color='w', label='Extrema-Pool idx', markerfacecolor=sparse_activation_color_list[3]),
            Line2D([0], [0], marker='o', color='w', label='Extrema', markerfacecolor=sparse_activation_color_list[4])
            ]
    fig_legend.legend(handles=legend_elements, fontsize=22, loc='upper center')
    plt.savefig(f'{path_images_mean_inverse_compression_ratio_vs_mean_reconstruction_loss_variable_kernel_size_list}/legend.pdf')
    plt.close()

    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    ax = plt.gca()
    for_density_plot = for_density_plot.reshape(for_density_plot.shape[0], -1, 2)
    nbins = 200
    yi, xi = np.mgrid[0:2.5:nbins*1j, 0:2.5:nbins*1j]
    for index, (sparse_activation, sparse_activation_name, sparse_activation_colormap, sparse_activation_color, c) in enumerate(zip(sparse_activation_list, sparse_activation_name_list, ['Blues', 'Oranges', 'Greens', 'Reds', 'Purples'], sparse_activation_color_list, for_density_plot)):
        k = gaussian_kde(c.T)
        zi = k(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
        plt.contourf(zi.reshape(xi.shape), [1, 999], colors=sparse_activation_color, extent=(0, 2.5, 0, 2.5))
    plt.axhspan(2, 2.5, alpha=0.3, color='r')
    plt.axhspan(1, 2, alpha=0.3, color='orange')
    plt.axvspan(1, 2.5, alpha=0.3, color='gray')
    wedge = patches.Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color='g')
    ax.add_patch(wedge)
    plt.xlabel(r'$\tilde{\mathcal{L}}$', fontsize=20)
    plt.ylabel(r'$CR^{-1}$', fontsize=20)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.grid(True)
    plt.savefig(f'{path_paper}/images_1d/crrl_density_plot.pdf')
    plt.close()

    print('UCI baseline, Supervised CNN classification')
    batch_size = 64
    lr = 0.01
    download_uci_epilepsy()
    training_dataset = UCIepilepsyDataset('training')
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    validation_dataset = UCIepilepsyDataset('validation')
    validation_dataloader = DataLoader(dataset=validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    test_dataset = UCIepilepsyDataset('test')
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
    sparse_activation_list = [identity_1d, relu_1d, topk_absolutes_1d, extrema_pool_indices_1d, extrema_1d]
    kernel_size_list_list = [2*[k] for k in uci_epilepsy_kernel_size_range]
    batch_size = 64
    lr = 0.01
    uci_epilepsy_supervised_latex_table = []
    training_dataset = UCIepilepsyDataset('training')
    training_dataloader = DataLoader(dataset=training_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(uci_epilepsy_training_range))
    validation_dataset = UCIepilepsyDataset('validation')
    validation_dataloader = DataLoader(dataset=validation_dataset, sampler=SubsetRandomSampler(uci_epilepsy_validation_range))
    test_dataset = UCIepilepsyDataset('test')
    test_dataloader = DataLoader(dataset=test_dataset, sampler=SubsetRandomSampler(uci_epilepsy_test_range))
    for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
        print(f'index_kernel_size_list: {index_kernel_size_list}')
        uci_epilepsy_supervised_latex_table_row = []
        for index_sparse_activation, sparse_activation in enumerate(sparse_activation_list):
            if sparse_activation == extrema_1d:
                sparsity_density_list = np.clip([k - 2 for k in kernel_size_list], 1, 999).tolist()
            else:
                sparsity_density_list = kernel_size_list
            mean_flithos_epoch_best = float('inf')
            model = SAN1d(sparse_activation, kernel_size_list, sparsity_density_list, device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_unsupervised_model(model, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    model_epoch_best = model
                    mean_flithos_epoch_best = flithos_epoch.mean()
            for neuron in model.neuron_list:
                neuron.weights.requires_grad_(False)
            mean_flithos_epoch_best = float('inf')
            supervised_model = CNN(len(training_dataset.labels.unique())).to(device)
            optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_supervised_model(supervised_model, model_epoch_best, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model_epoch_best, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    supervised_model_best = supervised_model
                    model_best = model_epoch_best
                    mean_flithos_epoch_best = flithos_epoch.mean()
            flithos, inverse_compression_ratio, reconstruction_loss, accuracy = validate_or_test_supervised_model(supervised_model_best, model_best, test_dataloader, device)
            uci_epilepsy_supervised_latex_table_row.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - uci_epilepsy_supervised_accuracy])
            if kernel_size_list[0] == 10:
                save_images_1d(model_best, dataset_name, test_dataset[0][0][0], kernel_size_list[0], device)
        uci_epilepsy_supervised_latex_table.append(uci_epilepsy_supervised_latex_table_row)
    header = [r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$', r'A\textsubscript{$\pm$\%}']
    index = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    uci_epilepsy_supervised_latex_table = np.array(uci_epilepsy_supervised_latex_table).T.tolist()
    df = pd.DataFrame(uci_epilepsy_supervised_latex_table, index=index)
    df = df.T
    df.index = list(uci_epilepsy_kernel_size_range)
    df.index.names = [r'$m$']
    formatters = 5*[lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:+.1f}']
    df.to_latex(f'{path_tables}/uci_epilepsy_supervised.tex', bold_rows=True, escape=False, column_format='l|rrrr|rrrr|rrrr|rrrr|rrrr', multicolumn_format='c', formatters=formatters)

    print('MNIST baseline, Supervised FNN classification')
    batch_size = 64
    lr = 0.01
    training_validation_dataset = datasets.MNIST(path_dataset, download=True, train=True, transform=transforms.ToTensor())
    training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_training_range))
    validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(mnist_validation_range), batch_size=batch_size)
    test_dataset = datasets.MNIST(path_dataset, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(mnist_test_range))
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
    mnist_supervised_accuracy = 100 * correct / total

    print('MNIST, Supervised FNN classification')
    dataset_name = 'MNIST'
    sparse_activation_list = [identity_2d, relu_2d, topk_absolutes_2d, extrema_pool_indices_2d, extrema_2d]
    kernel_size_list_list = [2*[k] for k in mnist_kernel_size_range]
    batch_size = 64
    lr = 0.01
    mnist_supervised_latex_table = []
    training_validation_dataset = datasets.MNIST(path_dataset, download=True, train=True, transform=transforms.ToTensor())
    training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(mnist_training_range))
    validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(mnist_validation_range))
    test_dataset = datasets.MNIST(path_dataset, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(mnist_test_range))
    for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
        print(f'index_kernel_size_list: {index_kernel_size_list}')
        mnist_supervised_latex_table_row = []
        for index_sparse_activation, sparse_activation in enumerate(sparse_activation_list):
            if sparse_activation == extrema_2d:
                sparsity_density_list = np.clip([k - 2 for k in kernel_size_list], 1, 999).tolist()
            else:
                sparsity_density_list = kernel_size_list
            mean_flithos_epoch_best = float('inf')
            model = SAN2d(sparse_activation, kernel_size_list, sparsity_density_list, device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_unsupervised_model(model, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    model_epoch_best = model
                    mean_flithos_epoch_best = flithos_epoch.mean()
            for neuron in model.neuron_list:
                neuron.weights.requires_grad_(False)
            mean_flithos_epoch_best = float('inf')
            supervised_model = FNN(training_validation_dataset.data[0], len(training_validation_dataset.classes)).to(device)
            optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_supervised_model(supervised_model, model_epoch_best, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model_epoch_best, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    supervised_model_best = supervised_model
                    model_best = model_epoch_best
                    mean_flithos_epoch_best = flithos_epoch.mean()
            flithos, inverse_compression_ratio, reconstruction_loss, accuracy = validate_or_test_supervised_model(supervised_model_best, model_best, test_dataloader, device)
            mnist_supervised_latex_table_row.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - mnist_supervised_accuracy])
            if kernel_size_list[0] == 4:
                save_images_2d(model_best, test_dataset[0][0][0], dataset_name, device)
        mnist_supervised_latex_table.append(mnist_supervised_latex_table_row)
    header = [r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$', r'A\textsubscript{$\pm$\%}']
    index = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    mnist_supervised_latex_table = np.array(mnist_supervised_latex_table).T.tolist()
    df = pd.DataFrame(mnist_supervised_latex_table, index=index)
    df = df.T
    df.index = list(mnist_kernel_size_range)
    df.index.names = [r'$m$']
    formatters = 5*[lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:+.1f}']
    df.to_latex(f'{path_tables}/mnist_supervised.tex', bold_rows=True, escape=False, column_format='l|rrrr|rrrr|rrrr|rrrr|rrrr', multicolumn_format='c', formatters=formatters)

    print('FashionMNIST baseline, Supervised FNN classification')
    batch_size = 64
    lr = 0.01
    training_validation_dataset = datasets.FashionMNIST(path_dataset, download=True, train=True, transform=transforms.ToTensor())
    training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(fashionmnist_training_range))
    validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(fashionmnist_validation_range), batch_size=batch_size)
    test_dataset = datasets.FashionMNIST(path_dataset, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(fashionmnist_test_range))
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
    fashionmnist_supervised_accuracy = 100 * correct / total

    print('FashionMNIST, Supervised FNN classification')
    dataset_name = 'FashionMNIST'
    sparse_activation_list = [identity_2d, relu_2d, topk_absolutes_2d, extrema_pool_indices_2d, extrema_2d]
    kernel_size_list_list = [2*[k] for k in fashionmnist_kernel_size_range]
    batch_size = 64
    lr = 0.01
    fashionmnist_supervised_latex_table = []
    training_validation_dataset = datasets.FashionMNIST(path_dataset, download=True, train=True, transform=transforms.ToTensor())
    training_dataloader = DataLoader(training_validation_dataset, batch_size=batch_size, sampler=SubsetRandomSampler(fashionmnist_training_range))
    validation_dataloader = DataLoader(training_validation_dataset, sampler=SubsetRandomSampler(fashionmnist_validation_range))
    test_dataset = datasets.FashionMNIST(path_dataset, train=False, transform=transforms.ToTensor())
    test_dataloader = DataLoader(test_dataset, sampler=SubsetRandomSampler(fashionmnist_test_range))
    for index_kernel_size_list, kernel_size_list in enumerate(kernel_size_list_list):
        print(f'index_kernel_size_list: {index_kernel_size_list}')
        fashionmnist_supervised_latex_table_row = []
        for index_sparse_activation, sparse_activation in enumerate(sparse_activation_list):
            if sparse_activation == extrema_2d:
                sparsity_density_list = np.clip([k - 2 for k in kernel_size_list], 1, 999).tolist()
            else:
                sparsity_density_list = kernel_size_list
            mean_flithos_epoch_best = float('inf')
            model = SAN2d(sparse_activation, kernel_size_list, sparsity_density_list, device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_unsupervised_model(model, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    model_epoch_best = model
                    mean_flithos_epoch_best = flithos_epoch.mean()
            for neuron in model.neuron_list:
                neuron.weights.requires_grad_(False)
            mean_flithos_epoch_best = float('inf')
            supervised_model = FNN(training_validation_dataset.data[0], len(training_validation_dataset.classes)).to(device)
            optimizer = optim.Adam(supervised_model.parameters(), lr=lr)
            for epoch in range(num_epochs):
                train_supervised_model(supervised_model, model_epoch_best, optimizer, training_dataloader, device)
                flithos_epoch, *_ = validate_or_test_unsupervised_model(model_epoch_best, validation_dataloader, device)
                if flithos_epoch.mean() < mean_flithos_epoch_best:
                    supervised_model_best = supervised_model
                    model_best = model_epoch_best
                    mean_flithos_epoch_best = flithos_epoch.mean()
            flithos, inverse_compression_ratio, reconstruction_loss, accuracy = validate_or_test_supervised_model(supervised_model_best, model_best, test_dataloader, device)
            fashionmnist_supervised_latex_table_row.extend([inverse_compression_ratio.mean(), reconstruction_loss.mean(), flithos.mean(), accuracy - fashionmnist_supervised_accuracy])
            if kernel_size_list[0] == 3:
                save_images_2d(model_best, test_dataset[0][0][0], dataset_name, device)
        fashionmnist_supervised_latex_table.append(fashionmnist_supervised_latex_table_row)
    header = [r'$CR^{-1}$', r'$\tilde{\mathcal{L}}$', r'$\bar\varphi$', r'A\textsubscript{$\pm$\%}']
    index = pd.MultiIndex.from_product([sparse_activation_name_list, header])
    fashionmnist_supervised_latex_table = np.array(fashionmnist_supervised_latex_table).T.tolist()
    df = pd.DataFrame(fashionmnist_supervised_latex_table, index=index)
    df = df.T
    df.index = list(fashionmnist_kernel_size_range)
    df.index.names = [r'$m$']
    formatters = 5*[lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:.2f}', lambda x: f'{x:+.1f}']
    df.to_latex(f'{path_tables}/fashionmnist_supervised.tex', bold_rows=True, escape=False, column_format='l|rrrr|rrrr|rrrr|rrrr|rrrr', multicolumn_format='c', formatters=formatters)

    df = pd.DataFrame({'key': ['uci_epilepsy_supervised_accuracy', 'mnist_supervised_accuracy', 'fashionmnist_supervised_accuracy'], 'value': [uci_epilepsy_supervised_accuracy, mnist_supervised_accuracy, fashionmnist_supervised_accuracy]})
    df.to_csv(f'{path_paper}/keys_values.csv', index=False, float_format='%.2f')
