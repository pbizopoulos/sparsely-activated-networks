import matplotlib.pyplot as plt
import torch

from torch import nn

from sparsely_activated_networks_pytorch import _conv2d_same_padding

plt.rcParams['image.interpolation'] = 'none'
plt.rcParams['savefig.format'] = 'pdf'
plt.rcParams['savefig.bbox'] = 'tight'


def identity_2d(x, kernel_size):
    return x

def relu_2d(x, kernel_size):
    return torch.relu(x)

def save_images_2d(model, data, dataset_name, path_results):
    model = model.to('cpu')
    fig = plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data.cpu().detach().numpy(), cmap='twilight', vmin=-2, vmax=2)
    plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-signal')
    plt.close()

    model.eval()
    with torch.no_grad():
        reconstructed, activations_list = model(data.unsqueeze(0).unsqueeze(0))
        for index_weights, (weights, activations) in enumerate(zip(model.weights_list, activations_list[0, :, 0])):
            fig = plt.figure(figsize=(4.8/2, 4.8/2))
            plt.imshow(weights.flip(0).flip(1).cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(weights).max(), vmax=2*abs(weights).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-kernel-{index_weights}')
            plt.close()

            similarity = _conv2d_same_padding(data.unsqueeze(0).unsqueeze(0), weights)[0, 0]
            fig = plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(similarity.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(similarity).max(), vmax=2*abs(similarity).max())
            plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-similarity-{index_weights}')
            plt.close()

            fig = plt.figure()
            plt.imshow(activations.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(activations).max(), vmax=2*abs(activations).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-activations-{index_weights}')
            plt.close()

            reconstruction = _conv2d_same_padding(activations.unsqueeze(0).unsqueeze(0), weights)[0, 0]
            fig = plt.figure()
            plt.imshow(reconstruction.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstruction).max(), vmax=2*abs(reconstruction).max())
            plt.xticks([])
            plt.yticks([])
            plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-reconstruction-{index_weights}')
            plt.close()

        fig = plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstructed).max(), vmax=2*abs(reconstructed).max())
        plt.savefig(f'{path_results}/{dataset_name}-{model.sparse_activation.__name__.replace("_", "-")}-{len(model.weights_list)}-reconstructed')
        plt.close()


class FNN(nn.Module):
    def __init__(self, sample_data, num_classes):
        super(FNN, self).__init__()
        self.fc = nn.Linear(sample_data.shape[-1]*sample_data.shape[-2], num_classes)

    def forward(self, batch_x):
        x = batch_x.view(batch_x.shape[0], -1)
        out = self.fc(x)
        return out
