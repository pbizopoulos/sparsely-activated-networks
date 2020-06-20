import matplotlib.pyplot as plt
import os
import torch


def identity_2d(x, kernel_size):
    return x

def relu_2d(x, kernel_size):
    return torch.relu(x)

def save_images_2d(model, data, dataset_name, device, path_results):
    fig = plt.figure(constrained_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data.cpu().detach().numpy(), cmap='twilight', vmin=-2, vmax=2)
    plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_signal.pdf', bbox_inches='tight')
    plt.close()

    model.eval()
    reconstructed, activations_list = model(data.unsqueeze(0).unsqueeze(0).to(device))
    similarity_list = torch.zeros_like(activations_list)
    reconstructions = torch.zeros_like(activations_list)
    for index_weights, (weights, similarity, activations, reconstruction) in enumerate(zip(model.weights_list, similarity_list[0, :, 0], activations_list[0, :, 0], reconstructions[0, :, 0])):
        fig = plt.figure(constrained_layout=True, figsize=(6.4/2, 4.8/2))
        plt.imshow(weights.flip(0).flip(1).cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(weights).max(), vmax=2*abs(weights).max())
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_kernel_{index_weights}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        plt.xticks([])
        plt.yticks([])
        plt.imshow(similarity.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(similarity).max(), vmax=2*abs(similarity).max())
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_similarity_{index_weights}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        plt.imshow(activations.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(activations).max(), vmax=2*abs(activations).max())
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_activations_{index_weights}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        plt.imshow(reconstruction.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstruction).max(), vmax=2*abs(reconstruction).max())
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_reconstruction_{index_weights}.pdf', bbox_inches='tight')
        plt.close()

    fig = plt.figure(constrained_layout=True)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstructed).max(), vmax=2*abs(reconstructed).max())
    plt.savefig(f'{path_results}/{dataset_name}_{model.sparse_activation.__name__}_{len(model.weights_list)}_reconstructed.pdf', bbox_inches='tight')
    plt.close()
