import matplotlib.pyplot as plt
import os

from utilities import filename_format, path_paper


def save_images_2d(model, data, dataset_name, device):
    model.eval()
    path_images_2d = f'{path_paper}/images_2d'
    if not os.path.exists(path_images_2d):
        os.mkdir(path_images_2d)

    fig = plt.figure(constrained_layout=True)
    apply_plot_style_2d()
    plt.imshow(data.cpu().detach().numpy(), cmap='twilight', vmin=-2, vmax=2)
    plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_signal.pdf', bbox_inches='tight')
    plt.close()

    reconstructed, similarity_list, activations_list, reconstructions = model(data.unsqueeze(0).unsqueeze(0).to(device))
    for index_neuron, (neuron, similarity, activations, reconstruction) in enumerate(zip(model.neuron_list, similarity_list[0, :, 0], activations_list[0, :, 0], reconstructions[0, :, 0])):
        fig = plt.figure(constrained_layout=True, figsize=(6.4/2, 4.8/2))
        apply_plot_style_2d()
        plt.imshow(neuron.weights.flip(0).flip(1).cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(neuron.weights).max(), vmax=2*abs(neuron.weights).max())
        plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_kernel_{index_neuron}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        apply_plot_style_2d()
        plt.imshow(similarity.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(similarity).max(), vmax=2*abs(similarity).max())
        plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_similarity_{index_neuron}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        apply_plot_style_2d()
        plt.imshow(activations.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(activations).max(), vmax=2*abs(activations).max())
        plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_activations_{index_neuron}.pdf', bbox_inches='tight')
        plt.close()

        fig = plt.figure(constrained_layout=True)
        apply_plot_style_2d()
        plt.imshow(reconstruction.cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstruction).max(), vmax=2*abs(reconstruction).max())
        plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_reconstruction_{index_neuron}.pdf', bbox_inches='tight')
        plt.close()

    fig = plt.figure(constrained_layout=True)
    apply_plot_style_2d()
    plt.imshow(reconstructed[0, 0].cpu().detach().numpy(), cmap='twilight', vmin=-2*abs(reconstructed).max(), vmax=2*abs(reconstructed).max())
    plt.savefig(f'{filename_format(path_images_2d, dataset_name, model)}_reconstructed.pdf', bbox_inches='tight')
    plt.close()

def apply_plot_style_2d():
    plt.xticks([])
    plt.yticks([])
