import glob
from os import environ
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pandas as pd
import requests
import torch
import wfdb
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Wedge
from matplotlib.ticker import MaxNLocator
from scipy.stats import gaussian_kde
from torch import nn, optim
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.datasets import MNIST, FashionMNIST
from torchvision.transforms import ToTensor


class CNN(nn.Module):
    def __init__(self: "CNN", classes_num: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(1, 3, 5)
        self.conv2 = nn.Conv1d(3, 16, 5)
        self.fc1 = nn.Linear(656, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, classes_num)

    def forward(self: "CNN", input_: torch.Tensor) -> torch.Tensor:
        out = functional.relu(self.conv1(input_))
        out = functional.max_pool1d(out, 2)
        out = functional.relu(self.conv2(out))
        out = functional.max_pool1d(out, 2)
        out = out.view(out.size(0), -1)
        out = functional.relu(self.fc1(out))
        out = functional.relu(self.fc2(out))
        output: torch.Tensor = self.fc3(out)
        return output


def extrema_1d(input_: torch.Tensor, minimum_extrema_distance: int) -> torch.Tensor:
    extrema_primary = torch.zeros_like(input_)
    dx = input_[:, :, 1:] - input_[:, :, :-1]
    dx_padright_greater = functional.pad(dx, [0, 1]) > 0
    dx_padleft_less = functional.pad(dx, [1, 0]) <= 0
    sign = (1 - torch.sign(input_)).bool()
    valleys = dx_padright_greater & dx_padleft_less & sign
    peaks = ~dx_padright_greater & ~dx_padleft_less & ~sign
    extrema = peaks | valleys
    extrema.squeeze_(1)
    for index, (x_, e_) in enumerate(zip(input_, extrema, strict=True)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, descending=True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices][:, 0]
        extrema_is_secondary = torch.zeros_like(
            extrema_indices_indices,
            dtype=torch.bool,
        )
        for index_, extrema_index in enumerate(extrema_indices_sorted):
            if not extrema_is_secondary[index_]:
                extrema_indices_r = (
                    extrema_indices_sorted >= extrema_index - minimum_extrema_distance
                )
                extrema_indices_l = (
                    extrema_indices_sorted <= extrema_index + minimum_extrema_distance
                )
                extrema_indices_m = extrema_indices_r & extrema_indices_l
                extrema_is_secondary = extrema_is_secondary | extrema_indices_m
                extrema_is_secondary[index_] = False
        extrema_primary_indices = extrema_indices_sorted[~extrema_is_secondary]
        extrema_primary[index, :, extrema_primary_indices] = x_[
            0,
            extrema_primary_indices,
        ]
    return extrema_primary


class Extrema1D(nn.Module):
    def __init__(self: "Extrema1D", minimum_extrema_distance: int) -> None:
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self: "Extrema1D", input_: torch.Tensor) -> torch.Tensor:
        return extrema_1d(input_, self.minimum_extrema_distance)


def extrema_2d(
    input_: torch.Tensor,
    minimum_extrema_distance: list[int],
) -> torch.Tensor:
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
    for index, (x_, e_) in enumerate(zip(input_, extrema, strict=True)):
        extrema_indices = torch.nonzero(e_, as_tuple=False)
        extrema_indices_indices = torch.argsort(abs(x_[0, e_]), 0, descending=True)
        extrema_indices_sorted = extrema_indices[extrema_indices_indices]
        extrema_is_secondary = torch.zeros_like(
            extrema_indices_indices,
            dtype=torch.bool,
        )
        for index_, (extrema_index_x, extrema_index_y) in enumerate(
            extrema_indices_sorted,
        ):
            if not extrema_is_secondary[index_]:
                extrema_indices_r = (
                    extrema_indices_sorted[:, 0]
                    >= extrema_index_x - minimum_extrema_distance[0]
                )
                extrema_indices_l = (
                    extrema_indices_sorted[:, 0]
                    <= extrema_index_x + minimum_extrema_distance[0]
                )
                extrema_indices_t = (
                    extrema_indices_sorted[:, 1]
                    >= extrema_index_y - minimum_extrema_distance[1]
                )
                extrema_indices_b = (
                    extrema_indices_sorted[:, 1]
                    <= extrema_index_y + minimum_extrema_distance[1]
                )
                extrema_indices_m = (
                    extrema_indices_r
                    & extrema_indices_l
                    & extrema_indices_t
                    & extrema_indices_b
                )
                extrema_is_secondary = extrema_is_secondary | extrema_indices_m
                extrema_is_secondary[index_] = False
        extrema_primary_indices = extrema_indices_sorted[~extrema_is_secondary]
        for extrema_primary_index in extrema_primary_indices:
            extrema_primary[
                index,
                :,
                extrema_primary_index[0],
                extrema_primary_index[1],
            ] = x_[0, extrema_primary_index[0], extrema_primary_index[1]]
    return extrema_primary


class Extrema2D(nn.Module):
    def __init__(self: "Extrema2D", minimum_extrema_distance: list[int]) -> None:
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self: "Extrema2D", input_: torch.Tensor) -> torch.Tensor:
        return extrema_2d(input_, self.minimum_extrema_distance)


def extrema_pool_indices_1d(input_: torch.Tensor, kernel_size: int) -> torch.Tensor:
    extrema_primary = torch.zeros_like(input_)
    _, extrema_indices = functional.max_pool1d(
        abs(input_),
        kernel_size,
        return_indices=True,
    )
    return extrema_primary.scatter(
        -1,
        extrema_indices,
        input_.gather(-1, extrema_indices),
    )


class ExtremaPoolIndices1D(nn.Module):
    def __init__(self: "ExtremaPoolIndices1D", pool_size: int) -> None:
        super().__init__()
        self.pool_size = pool_size

    def forward(self: "ExtremaPoolIndices1D", input_: torch.Tensor) -> torch.Tensor:
        return extrema_pool_indices_1d(input_, self.pool_size)


def extrema_pool_indices_2d(input_: torch.Tensor, kernel_size: int) -> torch.Tensor:
    x_flattened = input_.view(input_.shape[0], -1)
    extrema_primary = torch.zeros_like(x_flattened)
    _, extrema_indices = functional.max_pool2d(
        abs(input_),
        kernel_size,
        return_indices=True,
    )
    return extrema_primary.scatter(
        -1,
        extrema_indices[..., 0, 0],
        x_flattened.gather(-1, extrema_indices[..., 0, 0]),
    ).view(input_.shape)


class ExtremaPoolIndices2D(nn.Module):
    def __init__(self: "ExtremaPoolIndices2D", pool_size: int) -> None:
        super().__init__()
        self.pool_size = pool_size

    def forward(self: "ExtremaPoolIndices2D", input_: torch.Tensor) -> torch.Tensor:
        return extrema_pool_indices_2d(input_, self.pool_size)


class FNN(nn.Module):
    def __init__(self: "FNN", classes_num: int, sample_data: torch.Tensor) -> None:
        super().__init__()
        self.fc = nn.Linear(sample_data.shape[-1] * sample_data.shape[-2], classes_num)

    def forward(self: "FNN", batch_x: torch.Tensor) -> torch.Tensor:
        batch_x = batch_x.view(batch_x.shape[0], -1)
        output: torch.Tensor = self.fc(batch_x)
        return output


class Hook:
    def __init__(self: "Hook", module: nn.Module) -> None:
        self.hook = module.register_forward_hook(self.hook_fn)  # type: ignore[arg-type]

    def hook_fn(
        self: "Hook",
        _: None,
        input_: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        self.input_ = input_
        self.output = output


class Identity1D(nn.Module):
    def __init__(self: "Identity1D", _: None) -> None:
        super().__init__()

    def forward(self: "Identity1D", input_: torch.Tensor) -> torch.Tensor:
        return input_


class Identity2D(nn.Module):
    def __init__(self: "Identity2D", _: None) -> None:
        super().__init__()

    def forward(self: "Identity2D", input_: torch.Tensor) -> torch.Tensor:
        return input_


class PhysionetDataset(Dataset):  # type: ignore[type-arg]
    def __init__(
        self: "PhysionetDataset",
        dataset_name: str,
        train_validation_test: str,
    ) -> None:
        dataset_path = Path("data") / dataset_name
        if not dataset_path.exists():
            record_name = wfdb.get_record_list(f"{dataset_name}/1.0.0")[0]
            wfdb.dl_database(
                dataset_name,
                dataset_path.as_posix(),
                records=[record_name],
                annotators=None,
            )
        files = glob.glob((dataset_path / "*.hea").as_posix())
        file_name = Path(files[0]).stem
        records = wfdb.rdrecord(dataset_path / file_name)
        data = torch.tensor(records.p_signal[:12000, 0], dtype=torch.float)
        if train_validation_test == "train":
            self.data = data[:6000]
        elif train_validation_test == "validation":
            self.data = data[6000:8000]
        elif train_validation_test == "test":
            self.data = data[8000:]
        self.data = self.data.reshape((-1, 1, 1000))

    def __getitem__(self: "PhysionetDataset", index: int) -> tuple[torch.Tensor, int]:
        out = self.data[index] - self.data[index].mean()
        out /= out.std()
        return (out, 0)

    def __len__(self: "PhysionetDataset") -> int:
        return self.data.shape[0]


class Relu1D(nn.Module):
    def __init__(self: "Relu1D", _: None) -> None:
        super().__init__()

    def forward(self: "Relu1D", input_: torch.Tensor) -> torch.Tensor:
        return functional.relu(input_)


class Relu2D(nn.Module):
    def __init__(self: "Relu2D", _: None) -> None:
        super().__init__()

    def forward(self: "Relu2D", input_: torch.Tensor) -> torch.Tensor:
        return functional.relu(input_)


class SAN1d(nn.Module):
    def __init__(
        self: "SAN1d",
        kernel_sizes: list[int],
        sparse_activations: list[nn.Module],
    ) -> None:
        super().__init__()
        self.sparse_activations = nn.ModuleList(sparse_activations)
        self.weights_kernels = nn.ParameterList(
            [
                nn.Parameter(0.1 * torch.ones(kernel_size))
                for kernel_size in kernel_sizes
            ],
        )

    def forward(self: "SAN1d", batch_x: torch.Tensor) -> torch.Tensor:
        reconstructions_sum = torch.zeros_like(batch_x)
        for sparse_activation, weights_kernel in zip(
            self.sparse_activations,
            self.weights_kernels,
            strict=True,
        ):
            similarity = functional.conv1d(
                batch_x,
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )
            activations = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + functional.conv1d(
                activations,
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )
        return reconstructions_sum


class SAN2d(nn.Module):
    def __init__(
        self: "SAN2d",
        kernel_sizes: list[int],
        sparse_activations: list[nn.Module],
    ) -> None:
        super().__init__()
        self.sparse_activations = nn.ModuleList(sparse_activations)
        self.weights_kernels = nn.ParameterList(
            [
                nn.Parameter(0.1 * torch.ones(kernel_size, kernel_size))
                for kernel_size in kernel_sizes
            ],
        )

    def forward(self: "SAN2d", batch_x: torch.Tensor) -> torch.Tensor:
        reconstructions_sum = torch.zeros_like(batch_x)
        for sparse_activation, weights_kernel in zip(
            self.sparse_activations,
            self.weights_kernels,
            strict=True,
        ):
            similarity = functional.conv2d(
                batch_x,
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )
            activations = sparse_activation(similarity)
            reconstructions_sum = reconstructions_sum + functional.conv2d(
                activations,
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )
        return reconstructions_sum


def topk_absolutes_1d(input_: torch.Tensor, topk: int) -> torch.Tensor:
    extrema_primary = torch.zeros_like(input_)
    _, extrema_indices = torch.topk(abs(input_), topk)
    return extrema_primary.scatter(
        -1,
        extrema_indices,
        input_.gather(-1, extrema_indices),
    )


class TopKAbsolutes1D(nn.Module):
    def __init__(self: "TopKAbsolutes1D", topk: int) -> None:
        super().__init__()
        self.topk = topk

    def forward(self: "TopKAbsolutes1D", input_: torch.Tensor) -> torch.Tensor:
        return topk_absolutes_1d(input_, self.topk)


def topk_absolutes_2d(input_: torch.Tensor, topk: int) -> torch.Tensor:
    x_flattened = input_.view(input_.shape[0], -1)
    extrema_primary = torch.zeros_like(x_flattened)
    _, extrema_indices = torch.topk(abs(x_flattened), topk)
    return extrema_primary.scatter(
        -1,
        extrema_indices,
        x_flattened.gather(-1, extrema_indices),
    ).view(input_.shape)


class TopKAbsolutes2D(nn.Module):
    def __init__(self: "TopKAbsolutes2D", topk: int) -> None:
        super().__init__()
        self.topk = topk

    def forward(self: "TopKAbsolutes2D", input_: torch.Tensor) -> torch.Tensor:
        return topk_absolutes_2d(input_, self.topk)


class UCIepilepsyDataset(Dataset):  # type: ignore[type-arg]
    def __init__(self: "UCIepilepsyDataset", train_validation_test: str) -> None:
        data_file_path = Path("data/data.csv")
        if not data_file_path.is_file():
            with data_file_path.open("wb") as file:
                response = requests.get(
                    "https://web.archive.org/web/20200318000445/http://archive.ics.uci.edu/ml/machine-learning-databases/00388/data.csv",
                    timeout=60,
                )
                file.write(response.content)
        dataset = pd.read_csv(data_file_path)
        dataset["y"] = dataset["y"].replace(3, 2)
        dataset["y"] = dataset["y"].replace(4, 3)
        dataset["y"] = dataset["y"].replace(5, 3)
        data_all = dataset.drop(columns=["Unnamed: 0", "y"])
        data_max = data_all.max().max()
        data_min = data_all.min().min()
        data_all = 2 * (data_all - data_min) / (data_max - data_min) - 1
        classes_all = dataset["y"]
        last_train_index = int(data_all.shape[0] * 0.76)
        last_validation_index = int(data_all.shape[0] * 0.88)
        if train_validation_test == "train":
            self.data = torch.tensor(
                data_all.to_numpy()[:last_train_index, :],
                dtype=torch.float,
            )
            self.classes = torch.tensor(classes_all[:last_train_index].to_numpy()) - 1
        elif train_validation_test == "validation":
            self.data = torch.tensor(
                data_all.to_numpy()[last_train_index:last_validation_index, :],
                dtype=torch.float,
            )
            self.classes = (
                torch.tensor(
                    classes_all[last_train_index:last_validation_index].to_numpy(),
                )
                - 1
            )
        elif train_validation_test == "test":
            self.data = torch.tensor(
                data_all.to_numpy()[last_validation_index:, :],
                dtype=torch.float,
            )
            self.classes = (
                torch.tensor(classes_all[last_validation_index:].to_numpy()) - 1
            )
        self.data.unsqueeze_(1)

    def __getitem__(
        self: "UCIepilepsyDataset",
        index: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return (self.data[index], self.classes[index])

    def __len__(self: "UCIepilepsyDataset") -> int:
        return self.classes.shape[0]


def calculate_inverse_compression_ratio(
    activations_num: npt.NDArray[np.float64],
    data: torch.Tensor,
    model: nn.Module,
) -> npt.NDArray[np.float64]:
    activation_multiplier = 1 + len(model.weights_kernels[0].shape)  # type: ignore[index] # noqa: E501
    parameters_num = sum(weights_kernel.shape[0] for weights_kernel in model.weights_kernels)  # type: ignore[union-attr] # noqa: E501
    return (activation_multiplier * activations_num + parameters_num) / (
        data.shape[-1] * data.shape[-2]
    )


def save_images_1d(  # noqa: PLR0915
    data: torch.Tensor,
    dataset_name: str,
    model: SAN1d,
    sparse_activation_name: str,
    xlim_weight: int,
) -> None:
    model = model.to("cpu")
    _, ax = plt.subplots()
    ax.tick_params(labelbottom=False, labelleft=False)
    plt.grid(visible=True)
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.plot(data.cpu().detach().numpy())
    plt.ylim([data.min(), data.max()])
    plt.savefig(
        f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-signal.png",
    )
    plt.close()
    hook_handles = [
        Hook(sparse_activation_) for sparse_activation_ in model.sparse_activations
    ]
    model.eval()
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_ = []
        for hook_handle in hook_handles:
            activations_.append(hook_handle.output)
        activations_stack = torch.stack(activations_, 1)
        for weights_index, (weights_kernel, activations) in enumerate(
            zip(model.weights_kernels, activations_stack[0, :, 0], strict=True),
        ):
            _, ax = plt.subplots(figsize=(2, 2.2))
            ax.tick_params(labelbottom=False, labelleft=False)
            ax.xaxis.get_offset_text().set_visible(False)  # noqa: FBT003
            ax.yaxis.get_offset_text().set_visible(False)  # noqa: FBT003
            plt.grid(visible=True)
            plt.autoscale(enable=True, axis="x", tight=True)
            plt.plot(weights_kernel.cpu().detach().numpy(), "r")
            plt.xlim([0, xlim_weight])
            if dataset_name == "apnea-ecg":
                plt.ylabel(sparse_activation_name, fontsize=20)
            if sparse_activation_name == "relu":
                plt.title(dataset_name, fontsize=20)
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-kernel-{weights_index}.png",
            )
            plt.close()
            similarity = functional.conv1d(
                data.unsqueeze(0).unsqueeze(0),
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )[0, 0]
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(visible=True)
            plt.autoscale(enable=True, axis="x", tight=True)
            plt.plot(similarity.cpu().detach().numpy(), "g")
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-similarity-{weights_index}.png",
            )
            plt.close()
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(visible=True)
            plt.autoscale(enable=True, axis="x", tight=True)
            peaks = torch.nonzero(activations, as_tuple=False)[:, 0]
            plt.plot(similarity.cpu().detach().numpy(), "g", alpha=0.5)
            if peaks.shape[0] != 0:
                plt.stem(
                    peaks.cpu().detach().numpy(),
                    activations[peaks.cpu().detach().numpy()].cpu().detach().numpy(),
                    linefmt="c",
                    basefmt=" ",
                )
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-activations-{weights_index}.png",
            )
            plt.close()
            reconstruction_ = functional.conv1d(
                activations.unsqueeze(0).unsqueeze(0),
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )[0, 0]
            _, ax = plt.subplots()
            ax.tick_params(labelbottom=False, labelleft=False)
            plt.grid(visible=True)
            plt.autoscale(enable=True, axis="x", tight=True)
            reconstruction = reconstruction_.cpu().detach().numpy()
            lefts = peaks - weights_kernel.shape[0] / 2
            rights = peaks + weights_kernel.shape[0] / 2
            if weights_kernel.shape[0] % 2 == 1:
                rights += 1
            step = np.zeros_like(reconstruction, dtype=bool)
            lefts[lefts < 0] = 0
            rights[rights > reconstruction.shape[0]] = reconstruction.shape[0]
            for left, right in zip(lefts, rights, strict=True):
                step[int(left) : int(right)] = True
            pos_signal = reconstruction.copy()
            neg_signal = reconstruction.copy()
            pos_signal[step] = np.nan
            neg_signal[~step] = np.nan
            plt.plot(pos_signal)
            plt.plot(neg_signal, color="r")
            plt.ylim([data.min(), data.max()])
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-reconstruction-{weights_index}.png",
            )
            plt.close()
        _, ax = plt.subplots()
        ax.tick_params(labelbottom=False, labelleft=False)
        plt.grid(visible=True)
        plt.autoscale(enable=True, axis="x", tight=True)
        plt.plot(data.cpu().detach().numpy(), alpha=0.5)
        plt.plot(reconstructed[0, 0].cpu().detach().numpy(), "r")
        plt.ylim([data.min(), data.max()])
        plt.savefig(
            f"bin/{dataset_name}-{sparse_activation_name}-1d-{len(model.weights_kernels)}-reconstructed.png",
        )
        plt.close()


def save_images_2d(
    data: torch.Tensor,
    dataset_name: str,
    model: SAN2d,
    sparse_activation_name: str,
) -> None:
    model = model.to("cpu")
    plt.figure()
    plt.xticks([])
    plt.yticks([])
    plt.imshow(data.cpu().detach().numpy(), cmap="twilight", vmin=-2, vmax=2)
    plt.savefig(
        f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-signal.png",
    )
    plt.close()
    hook_handles = [
        Hook(sparse_activation_) for sparse_activation_ in model.sparse_activations
    ]
    model.eval()
    with torch.no_grad():
        reconstructed = model(data.unsqueeze(0).unsqueeze(0))
        activations_ = []
        for hook_handle in hook_handles:
            activations_.append(hook_handle.output)
        activations_stack = torch.stack(activations_, 1)
        for weights_index, (weights_kernel, activations) in enumerate(
            zip(model.weights_kernels, activations_stack[0, :, 0], strict=True),
        ):
            plt.figure(figsize=(4.8 / 2, 4.8 / 2))
            plt.imshow(
                weights_kernel.flip(0).flip(1).cpu().detach().numpy(),
                cmap="twilight",
                vmin=-2 * abs(weights_kernel).max(),
                vmax=2 * abs(weights_kernel).max(),
            )
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-kernel-{weights_index}.png",
            )
            plt.close()
            similarity = functional.conv2d(
                data.unsqueeze(0).unsqueeze(0),
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )[0, 0]
            plt.figure()
            plt.xticks([])
            plt.yticks([])
            plt.imshow(
                similarity.cpu().detach().numpy(),
                cmap="twilight",
                vmin=-2 * abs(similarity).max(),
                vmax=2 * abs(similarity).max(),
            )
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-similarity-{weights_index}.png",
            )
            plt.close()
            plt.figure()
            plt.imshow(
                activations.cpu().detach().numpy(),
                cmap="twilight",
                vmin=-2 * abs(activations).max(),
                vmax=2 * abs(activations).max(),
            )
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-activations-{weights_index}.png",
            )
            plt.close()
            reconstruction = functional.conv2d(
                activations.unsqueeze(0).unsqueeze(0),
                weights_kernel.unsqueeze(0).unsqueeze(0),
                padding="same",
            )[0, 0]
            plt.figure()
            plt.imshow(
                reconstruction.cpu().detach().numpy(),
                cmap="twilight",
                vmin=-2 * abs(reconstruction).max(),
                vmax=2 * abs(reconstruction).max(),
            )
            plt.xticks([])
            plt.yticks([])
            plt.savefig(
                f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-reconstruction-{weights_index}.png",
            )
            plt.close()
        plt.figure()
        plt.xticks([])
        plt.yticks([])
        plt.imshow(
            reconstructed[0, 0].cpu().detach().numpy(),
            cmap="twilight",
            vmin=-2 * abs(reconstructed).max(),
            vmax=2 * abs(reconstructed).max(),
        )
        plt.savefig(
            f"bin/{dataset_name}-{sparse_activation_name}-2d-{len(model.weights_kernels)}-reconstructed.png",
        )
        plt.close()


def train_model_supervised(
    dataloader_train: DataLoader[int],
    model_supervised: nn.Module,
    model_unsupervised: nn.Module,
    optimizer: optim.Adam,
) -> None:
    device = next(model_supervised.parameters()).device
    model_supervised.train()
    for data, target in dataloader_train:
        data = data.to(device)  # noqa: PLW2901
        target = target.to(device)  # noqa: PLW2901
        data_reconstructed = model_unsupervised(data)
        output = model_supervised(data_reconstructed)
        classification_loss = functional.cross_entropy(output, target)
        optimizer.zero_grad()
        classification_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def train_model_unsupervised(
    dataloader_train: DataLoader[int],
    model: nn.Module,
    optimizer: optim.Adam,
) -> None:
    device = next(model.parameters()).device
    model.train()
    for data, _ in dataloader_train:
        data = data.to(device)  # noqa: PLW2901
        data_reconstructed = model(data)
        reconstruction_loss = functional.l1_loss(data, data_reconstructed)
        optimizer.zero_grad()
        reconstruction_loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()


def validate_or_test_model_supervised(dataloader: DataLoader[int], hook_handles: list[Hook], model_supervised: nn.Module, model_unsupervised: nn.Module) -> tuple:  # type: ignore[type-arg] # noqa: E501
    device = next(model_supervised.parameters()).device
    predictions_correct_num = 0
    activations_num = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    model_supervised.eval()
    with torch.no_grad():
        for index, (data, target) in enumerate(dataloader):
            data = data.to(device)  # noqa: PLW2901
            target = target.to(device)  # noqa: PLW2901
            data_reconstructed = model_unsupervised(data)
            activations_ = []
            for hook_handle in hook_handles:
                activations_.append(hook_handle.output)
            activations_stack = torch.stack(activations_, 1)
            reconstruction_loss[index] = functional.l1_loss(
                data,
                data_reconstructed,
            ) / functional.l1_loss(data, torch.zeros_like(data))
            activations_num[index] = torch.nonzero(
                activations_stack,
                as_tuple=False,
            ).shape[0]
            output = model_supervised(data_reconstructed)
            prediction = output.argmax(dim=1)
            predictions_correct_num += sum(prediction == target).item()
    inverse_compression_ratio = calculate_inverse_compression_ratio(
        activations_num,
        data,
        model_unsupervised,
    )
    flithos = np.mean(
        [
            np.sqrt(icr**2 + rl**2)
            for icr, rl in zip(
                inverse_compression_ratio,
                reconstruction_loss,
                strict=True,
            )
        ],
    )
    return (
        flithos,
        inverse_compression_ratio,
        reconstruction_loss,
        100 * predictions_correct_num / len(dataloader),
    )


def validate_or_test_model_unsupervised(dataloader: DataLoader[int], hook_handles: list[Hook], model: nn.Module) -> tuple:  # type: ignore[type-arg] # noqa: E501
    device = next(model.parameters()).device
    activations_num = np.zeros(len(dataloader))
    reconstruction_loss = np.zeros(len(dataloader))
    model.eval()
    with torch.no_grad():
        for index, (data, _) in enumerate(dataloader):
            data = data.to(device)  # noqa: PLW2901
            data_reconstructed = model(data)
            activations_ = []
            for hook_handle in hook_handles:
                activations_.append(hook_handle.output)
            activations_stack = torch.stack(activations_, 1)
            reconstruction_loss[index] = functional.l1_loss(
                data,
                data_reconstructed,
            ) / functional.l1_loss(data, torch.zeros_like(data))
            activations_num[index] = torch.nonzero(
                activations_stack,
                as_tuple=False,
            ).shape[0]
    inverse_compression_ratio = calculate_inverse_compression_ratio(
        activations_num,
        data,
        model,
    )
    flithos = np.mean(
        [
            np.sqrt(icr**2 + rl**2)
            for icr, rl in zip(
                inverse_compression_ratio,
                reconstruction_loss,
                strict=True,
            )
        ],
    )
    return (flithos, inverse_compression_ratio, reconstruction_loss)


def main() -> None:  # noqa: C901, PLR0912, PLR0915
    bin_file_path = Path("bin")
    if not bin_file_path.exists():
        bin_file_path.mkdir(parents=True)
    data_file_path = Path("data")
    if not data_file_path.exists():
        data_file_path.mkdir(parents=True)
    plt.rcParams["font.size"] = 20
    plt.rcParams["image.interpolation"] = "none"
    plt.rcParams["savefig.bbox"] = "tight"
    epochs_physionet_num = 30
    epochs_num = 5
    kernel_size_physionet_range = range(1, 250)
    uci_epilepsy_train_range = range(8740)
    uci_epilepsy_validation_range = range(1380)
    uci_epilepsy_test_range = range(1380)
    mnist_fashionmnist_train_ranges = [range(50000), range(50000)]
    mnist_fashionmnist_validation_ranges = [range(50000, 60000), range(50000, 60000)]
    mnist_fashionmnist_test_ranges = [range(10000), range(10000)]
    if environ["DEBUG"] == "1":
        epochs_physionet_num = 3
        epochs_num = 2
        kernel_size_physionet_range = range(1, 10)
        uci_epilepsy_train_range = range(10)
        uci_epilepsy_validation_range = range(10)
        uci_epilepsy_test_range = range(10)
        mnist_fashionmnist_train_ranges = [range(10), range(10)]
        mnist_fashionmnist_validation_ranges = [range(10, 20), range(10, 20)]
        mnist_fashionmnist_test_ranges = [range(10), range(10)]
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    sparse_activation_names = [
        "Identity",
        "ReLU",
        "top-k absolutes",
        "Extrema-Pool indices",
        "Extrema",
    ]
    kernel_size_uci_epilepsy_range = range(8, 16)
    kernel_size_mnist_fashionmnist_ranges = [range(1, 7), range(1, 7)]
    dataset_names = [
        "apnea-ecg",
        "bidmc",
        "bpssrat",
        "cebsdb",
        "ctu-uhb-ctgdb",
        "drivedb",
        "emgdb",
        "mitdb",
        "noneeg",
        "prcp",
        "shhpsgdb",
        "slpdb",
        "sufhsdb",
        "voiced",
        "wrist",
    ]
    xlim_weights = [74, 113, 10, 71, 45, 20, 9, 229, 37, 105, 15, 232, 40, 70, 173]
    sparse_activations = [
        Identity1D,
        Relu1D,
        TopKAbsolutes1D,
        ExtremaPoolIndices1D,
        Extrema1D,
    ]
    kernel_sizes_list = [
        [kernel_size_physionet] for kernel_size_physionet in kernel_size_physionet_range
    ]
    batch_size = 2
    lr = 0.01
    sparse_activation_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][
        : len(sparse_activations)
    ]
    results_physionet_rows = []
    flithos_mean_array = np.zeros(
        (len(sparse_activations), len(dataset_names), len(kernel_sizes_list)),
    )
    flithos_all_validation_array = np.zeros(
        (
            len(sparse_activations),
            len(dataset_names),
            len(kernel_sizes_list),
            epochs_physionet_num,
        ),
    )
    kernel_size_best_array = np.zeros(
        (len(sparse_activations), len(dataset_names)),
        dtype=int,
    )
    gaussian_kde_input_array = np.zeros(
        (len(sparse_activations), len(dataset_names), len(kernel_sizes_list), 2),
    )
    for dataset_name_index, (dataset_name, xlim_weight) in enumerate(
        zip(dataset_names, xlim_weights, strict=True),
    ):
        results_physionet_row = []
        physionet_dataset_train = PhysionetDataset(dataset_name, "train")
        dataloader_train = DataLoader(
            dataset=physionet_dataset_train,
            batch_size=batch_size,
            shuffle=True,
        )
        physionet_dataset_validation = PhysionetDataset(dataset_name, "validation")
        dataloader_validation = DataLoader(dataset=physionet_dataset_validation)
        physionet_dataset_test = PhysionetDataset(dataset_name, "test")
        dataloader_test = DataLoader(dataset=physionet_dataset_test)
        fig, ax_main = plt.subplots(constrained_layout=True, figsize=(6, 6))
        for sparse_activation_index, (
            sparse_activation,
            sparse_activation_color,
            sparse_activation_name,
        ) in enumerate(
            zip(
                sparse_activations,
                sparse_activation_colors,
                sparse_activation_names,
                strict=True,
            ),
        ):
            flithos_mean_best = float("inf")
            for kernel_size_list_index, kernel_sizes in enumerate(kernel_sizes_list):
                flithos_epoch_mean_best = float("inf")
                if sparse_activation == TopKAbsolutes1D:
                    sparsity_densities = [
                        int(physionet_dataset_test.data.shape[-1] / kernel_size)
                        for kernel_size in kernel_sizes
                    ]
                elif sparse_activation == Extrema1D:
                    sparsity_densities = np.clip(
                        [kernel_size - 3 for kernel_size in kernel_sizes],
                        1,
                        999,
                    ).tolist()
                else:
                    sparsity_densities = kernel_sizes
                sparse_activation_list = [
                    sparse_activation(sparsity_density)
                    for sparsity_density in sparsity_densities
                ]
                san1d_model = SAN1d(kernel_sizes, sparse_activation_list).to(device)
                optimizer = optim.Adam(san1d_model.parameters(), lr=lr)
                hook_handles = [
                    Hook(sparse_activation_)
                    for sparse_activation_ in san1d_model.sparse_activations
                ]
                for epoch in range(epochs_physionet_num):
                    train_model_unsupervised(dataloader_train, san1d_model, optimizer)
                    flithos_epoch, *_ = validate_or_test_model_unsupervised(
                        dataloader_validation,
                        hook_handles,
                        san1d_model,
                    )
                    flithos_all_validation_array[
                        sparse_activation_index,
                        dataset_name_index,
                        kernel_size_list_index,
                        epoch,
                    ] = flithos_epoch.mean()
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        model_epoch_best = san1d_model
                        flithos_epoch_mean_best = flithos_epoch.mean()
                (
                    flithos_epoch_best,
                    inverse_compression_ratio_epoch_best,
                    reconstruction_loss_epoch_best,
                ) = validate_or_test_model_unsupervised(
                    dataloader_test,
                    hook_handles,
                    model_epoch_best,
                )
                flithos_mean_array[
                    sparse_activation_index,
                    dataset_name_index,
                    kernel_size_list_index,
                ] = flithos_epoch_best.mean()
                plt.sca(ax_main)
                plt.plot(
                    reconstruction_loss_epoch_best.mean(),
                    inverse_compression_ratio_epoch_best.mean(),
                    "o",
                    c=sparse_activation_color,
                    markersize=3,
                )
                gaussian_kde_input_array[
                    sparse_activation_index,
                    dataset_name_index,
                    kernel_size_list_index,
                ] = [
                    reconstruction_loss_epoch_best.mean(),
                    inverse_compression_ratio_epoch_best.mean(),
                ]
                if (
                    flithos_mean_array[
                        sparse_activation_index,
                        dataset_name_index,
                        kernel_size_list_index,
                    ]
                    < flithos_mean_best
                ):
                    kernel_size_best_array[
                        sparse_activation_index,
                        dataset_name_index,
                    ] = kernel_sizes[0]
                    inverse_compression_ratio_best = (
                        inverse_compression_ratio_epoch_best
                    )
                    reconstruction_loss_best = reconstruction_loss_epoch_best
                    flithos_mean_best = flithos_mean_array[
                        sparse_activation_index,
                        dataset_name_index,
                        kernel_size_list_index,
                    ]
                    model_best = model_epoch_best
            results_physionet_row.extend(
                [
                    kernel_size_best_array[sparse_activation_index, dataset_name_index],
                    inverse_compression_ratio_best.mean(),
                    reconstruction_loss_best.mean(),
                    flithos_mean_best,
                ],
            )
            save_images_1d(
                physionet_dataset_test[0][0][0],
                dataset_name,
                model_best,
                sparse_activation_name.lower().replace(" ", "-"),
                xlim_weight,
            )
            ax_main.arrow(
                reconstruction_loss_best.mean(),
                inverse_compression_ratio_best.mean(),
                1.83 - reconstruction_loss_best.mean(),
                2.25
                - 0.5 * sparse_activation_index
                - inverse_compression_ratio_best.mean(),
            )
            fig.add_axes(
                [0.75, 0.81 - 0.165 * sparse_activation_index, 0.1, 0.1],
                facecolor="y",
            )
            plt.plot(
                model_best.weights_kernels[0].flip(0).cpu().detach().numpy().T,
                c=sparse_activation_color,
            )
            plt.xlim([0, xlim_weight])
            plt.xticks([])
            plt.yticks([])
        results_physionet_rows.append(results_physionet_row)
        plt.sca(ax_main)
        plt.xlim([0, 2.5])
        plt.ylim([0, 2.5])
        plt.xlabel("$\\tilde{\\mathcal{L}}$")
        plt.ylabel("$CR^{-1}$")
        plt.grid(visible=True)
        plt.title(dataset_name)
        plt.axhspan(2, 2.5, alpha=0.3, color="r")
        plt.axhspan(1, 2, alpha=0.3, color="orange")
        plt.axvspan(1, 2.5, alpha=0.3, color="gray")
        wedge = Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color="g")
        ax_main.add_patch(wedge)
        plt.savefig(f"bin/{dataset_name}.png")
        plt.close()
    header = ["$m$", "$CR^{-1}$", "$\\tilde{\\mathcal{L}}$", "$\\bar\\varphi$"]
    columns = pd.MultiIndex.from_product([sparse_activation_names, header])
    results_physionet_df = pd.DataFrame(
        results_physionet_rows,
        columns=columns,
        index=dataset_names,
    )
    results_physionet_df.index.names = ["Datasets"]
    styler = results_physionet_df.style
    styler.format(
        precision=2,
        formatter={
            columns[0]: "{:.0f}",
            columns[4]: "{:.0f}",
            columns[8]: "{:.0f}",
            columns[12]: "{:.0f}",
            columns[16]: "{:.0f}",
        },
    )
    styler.to_latex(
        "bin/table-flithos-variable-kernel-size.tex",
        hrules=True,
        multicol_align="c",
    )
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    var = np.zeros((len(dataset_names), epochs_physionet_num))
    p1 = []
    p2 = []
    for (
        sparse_activation_color,
        kernel_size_best,
        flithos_all_validation_element_array,
    ) in zip(
        sparse_activation_colors,
        kernel_size_best_array,
        flithos_all_validation_array,
        strict=True,
    ):
        t_range = range(1, flithos_all_validation_element_array.shape[-1] + 1)
        for index_, (c_, k_) in enumerate(
            zip(flithos_all_validation_element_array, kernel_size_best, strict=True),
        ):
            var[index_] = c_[k_ - 1]
        mu = var.mean(axis=0)
        sigma = var.std(axis=0)
        ax.fill_between(
            t_range,
            mu + sigma,
            mu - sigma,
            facecolor=sparse_activation_color,
            alpha=0.3,
        )
        p1.append(ax.plot(t_range, mu, color=sparse_activation_color)[0])
        p2.append(ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)[0])
    ax.legend(
        [
            (p2[0], p1[0]),
            (p2[1], p1[1]),
            (p2[2], p1[2]),
            (p2[3], p1[3]),
            (p2[4], p1[4]),
        ],
        sparse_activation_names,
        fontsize=12,
        loc="lower left",
    )
    plt.xlabel("epochs")
    plt.ylabel("$\\bar\\varphi$")
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.ylim([0, 2.5])
    plt.grid(visible=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("bin/mean-flithos-validation-epochs.png")
    plt.close()
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    p1 = []
    p2 = []
    for sparse_activation_color, flithos_mean_element_array in zip(
        sparse_activation_colors,
        flithos_mean_array,
        strict=True,
    ):
        t_range = range(1, flithos_mean_element_array.shape[1] + 1)
        mu = flithos_mean_element_array.mean(axis=0)
        sigma = flithos_mean_element_array.std(axis=0)
        ax.fill_between(
            t_range,
            mu + sigma,
            mu - sigma,
            facecolor=sparse_activation_color,
            alpha=0.3,
        )
        p1.append(ax.plot(t_range, mu, color=sparse_activation_color)[0])
        p2.append(ax.fill(np.NaN, np.NaN, sparse_activation_color, alpha=0.3)[0])
    ax.legend(
        [
            (p2[0], p1[0]),
            (p2[1], p1[1]),
            (p2[2], p1[2]),
            (p2[3], p1[3]),
            (p2[4], p1[4]),
        ],
        sparse_activation_names,
        fontsize=12,
        loc="lower right",
    )
    plt.xlabel("$m$")
    plt.ylabel("$\\bar\\varphi$")
    plt.autoscale(enable=True, axis="x", tight=True)
    plt.ylim([0, 2.5])
    plt.grid(visible=True)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.savefig("bin/mean-flithos-variable-kernel-size-list.png")
    plt.close()
    fig = plt.figure(constrained_layout=True, figsize=(6, 6))
    fig_legend = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(range(2), range(2), range(2))
    non_sparse_model_description_patch = Patch(
        color="r",
        alpha=0.3,
        label="non-sparse model description",
    )
    worse_cr_than_original_data_patch = Patch(
        color="orange",
        alpha=0.3,
        label="worse $CR^{-1}$ than original data",
    )
    worse_l_than_constant_prediction_patch = Patch(
        color="gray",
        alpha=0.3,
        label="worse $\\tilde{\\mathcal{L}}$ than constant prediction",
    )
    varphi_less_than_one_patch = Patch(
        color="g",
        alpha=0.3,
        label="$\\bar\\varphi < 1$",
    )
    line2d_list = []
    for sparse_activation_name, sparse_activation_color in zip(
        sparse_activation_names,
        sparse_activation_colors,
        strict=True,
    ):
        line2d_list.append(
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=sparse_activation_name,
                markerfacecolor=sparse_activation_color,
            ),
        )
    legend_handle_list = [
        non_sparse_model_description_patch,
        worse_cr_than_original_data_patch,
        worse_l_than_constant_prediction_patch,
        varphi_less_than_one_patch,
        line2d_list[0],
        line2d_list[1],
        line2d_list[2],
        line2d_list[3],
        line2d_list[4],
    ]
    fig_legend.legend(handles=legend_handle_list, fontsize=22, loc="upper center")
    plt.savefig("bin/legend.png")
    plt.close()
    fig, ax = plt.subplots(constrained_layout=True, figsize=(6, 6))
    gaussian_kde_input_array = gaussian_kde_input_array.reshape(
        gaussian_kde_input_array.shape[0],
        -1,
        2,
    )
    nbins = 200
    yi, xi = np.mgrid[0 : 2.5 : nbins * 1j, 0 : 2.5 : nbins * 1j]  # type: ignore[misc]
    for sparse_activation_color, gaussian_kde_input_element_array in zip(
        sparse_activation_colors,
        gaussian_kde_input_array,
        strict=True,
    ):
        gkde = gaussian_kde(gaussian_kde_input_element_array.T)
        zi = gkde(np.vstack([xi.flatten(), yi.flatten()]))
        plt.contour(
            zi.reshape(xi.shape),
            [1, 999],
            colors=sparse_activation_color,
            extent=(0, 2.5, 0, 2.5),
        )
        plt.contourf(
            zi.reshape(xi.shape),
            [1, 999],
            colors=sparse_activation_color,
            extent=(0, 2.5, 0, 2.5),
        )
    plt.axhspan(2, 2.5, alpha=0.3, color="r")
    plt.axhspan(1, 2, alpha=0.3, color="orange")
    plt.axvspan(1, 2.5, alpha=0.3, color="gray")
    wedge = Wedge((0, 0), 1, theta1=0, theta2=90, alpha=0.3, color="g")
    ax.add_patch(wedge)
    plt.xlabel("$\\tilde{\\mathcal{L}}$")
    plt.ylabel("$CR^{-1}$")
    plt.xlim([0, 2.5])
    plt.ylim([0, 2.5])
    plt.grid(visible=True)
    plt.savefig("bin/crrl-density-plot.png")
    plt.close()
    batch_size = 64
    lr = 0.01
    uci_epilepsy_dataset_train = UCIepilepsyDataset("train")
    dataloader_train = DataLoader(
        dataset=uci_epilepsy_dataset_train,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(uci_epilepsy_train_range),
    )
    uci_epilepsy_dataset_validation = UCIepilepsyDataset("validation")
    dataloader_validation = DataLoader(
        dataset=uci_epilepsy_dataset_validation,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(uci_epilepsy_validation_range),
    )
    uci_epilepsy_dataset_test = UCIepilepsyDataset("test")
    dataloader_test = DataLoader(
        dataset=uci_epilepsy_dataset_test,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(uci_epilepsy_test_range),
    )
    accuracy_best = 0.0
    num_classes = len(uci_epilepsy_dataset_train.classes.unique())  # type: ignore[no-untyped-call] # noqa: E501
    model_supervised = CNN(num_classes).to(device)
    optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
    for _ in range(epochs_num):
        model_supervised.train()
        for data, target in dataloader_train:
            data = data.to(device)  # noqa: PLW2901
            target = target.to(device)  # noqa: PLW2901
            output = model_supervised(data)
            classification_loss = functional.cross_entropy(output, target)
            optimizer.zero_grad()
            classification_loss.backward()  # type: ignore[no-untyped-call]
            optimizer.step()
        predictions_correct_num = 0
        predictions_num = 0
        model_supervised.eval()
        with torch.no_grad():
            for data, target in dataloader_validation:
                data = data.to(device)  # noqa: PLW2901
                target = target.to(device)  # noqa: PLW2901
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
        for data, target in dataloader_test:
            data = data.to(device)  # noqa: PLW2901
            target = target.to(device)  # noqa: PLW2901
            output = model_supervised_best(data)
            prediction = output.argmax(dim=1)
            predictions_correct_num += sum(prediction == target).item()
            predictions_num += output.shape[0]
    accuracy_uci_epilepsy = 100 * predictions_correct_num / predictions_num
    dataset_name = "UCI-epilepsy"
    sparse_activations = [
        Identity1D,
        Relu1D,
        TopKAbsolutes1D,
        ExtremaPoolIndices1D,
        Extrema1D,
    ]
    kernel_sizes_list = [
        2 * [kernel_size_uci_epilepsy]
        for kernel_size_uci_epilepsy in kernel_size_uci_epilepsy_range
    ]
    batch_size = 64
    lr = 0.01
    results_supervised_rows_list = []
    uci_epilepsy_dataset_train = UCIepilepsyDataset("train")
    dataloader_train = DataLoader(
        dataset=uci_epilepsy_dataset_train,
        batch_size=batch_size,
        sampler=SubsetRandomSampler(uci_epilepsy_train_range),
    )
    uci_epilepsy_dataset_validation = UCIepilepsyDataset("validation")
    dataloader_validation = DataLoader(
        dataset=uci_epilepsy_dataset_validation,
        sampler=SubsetRandomSampler(uci_epilepsy_validation_range),
    )
    uci_epilepsy_dataset_test = UCIepilepsyDataset("test")
    dataloader_test = DataLoader(
        dataset=uci_epilepsy_dataset_test,
        sampler=SubsetRandomSampler(uci_epilepsy_test_range),
    )
    for kernel_sizes in kernel_sizes_list:
        results_supervised_rows = []
        for sparse_activation, sparse_activation_name in zip(
            sparse_activations,
            sparse_activation_names,
            strict=True,
        ):
            if sparse_activation == TopKAbsolutes1D:
                sparsity_densities = [
                    int(uci_epilepsy_dataset_test.data.shape[-1] / kernel_size)
                    for kernel_size in kernel_sizes
                ]
            elif sparse_activation == Extrema1D:
                sparsity_densities = np.clip(
                    [kernel_size - 2 for kernel_size in kernel_sizes],
                    1,
                    999,
                ).tolist()
            else:
                sparsity_densities = kernel_sizes
            sparse_activation_list = [
                sparse_activation(sparsity_density)
                for sparsity_density in sparsity_densities
            ]
            san1d_model = SAN1d(kernel_sizes, sparse_activation_list).to(device)
            optimizer = optim.Adam(san1d_model.parameters(), lr=lr)
            hook_handles = [
                Hook(sparse_activation_)
                for sparse_activation_ in san1d_model.sparse_activations
            ]
            flithos_epoch_mean_best = float("inf")
            for _ in range(epochs_num):
                train_model_unsupervised(dataloader_train, san1d_model, optimizer)
                flithos_epoch, *_ = validate_or_test_model_unsupervised(
                    dataloader_validation,
                    hook_handles,
                    san1d_model,
                )
                if flithos_epoch.mean() < flithos_epoch_mean_best:
                    model_epoch_best = san1d_model
                    flithos_epoch_mean_best = flithos_epoch.mean()
            for weights_kernel in san1d_model.weights_kernels:
                weights_kernel.requires_grad_(False)  # noqa: FBT003
            flithos_epoch_mean_best = float("inf")
            model_supervised = CNN(num_classes).to(device).to(device)
            optimizer = optim.Adam(model_supervised.parameters(), lr=lr)
            for _ in range(epochs_num):
                train_model_supervised(
                    dataloader_train,
                    model_supervised,
                    model_epoch_best,
                    optimizer,
                )
                flithos_epoch, *_ = validate_or_test_model_unsupervised(
                    dataloader_validation,
                    hook_handles,
                    model_epoch_best,
                )
                if flithos_epoch.mean() < flithos_epoch_mean_best:
                    model_supervised_best = model_supervised
                    model_best = model_epoch_best
                    flithos_epoch_mean_best = flithos_epoch.mean()
            (
                flithos,
                inverse_compression_ratio,
                reconstruction_loss,
                accuracy,
            ) = validate_or_test_model_supervised(
                dataloader_test,
                hook_handles,
                model_supervised_best,
                model_best,
            )
            results_supervised_rows.extend(
                [
                    inverse_compression_ratio.mean(),
                    reconstruction_loss.mean(),
                    flithos.mean(),
                    accuracy - accuracy_uci_epilepsy,
                ],
            )
            if kernel_sizes[0] == 10:  # noqa: PLR2004
                save_images_1d(
                    uci_epilepsy_dataset_test[0][0][0],
                    dataset_name,
                    model_best,
                    sparse_activation_name.lower().replace(" ", "-"),
                    kernel_sizes[0],
                )
        results_supervised_rows_list.append(results_supervised_rows)
    header = [
        "$CR^{-1}$",
        "$\\tilde{\\mathcal{L}}$",
        "$\\bar\\varphi$",
        "A\\textsubscript{$\\pm$\\%}",
    ]
    columns = pd.MultiIndex.from_product([sparse_activation_names, header])
    results_supervised_df = pd.DataFrame(
        results_supervised_rows_list,
        columns=columns,
        index=kernel_size_uci_epilepsy_range,
    )
    results_supervised_df.index.names = ["$m$"]
    styler = results_supervised_df.style
    styler.format(
        precision=2,
        formatter={
            columns[3]: "{:.1f}",
            columns[7]: "{:.1f}",
            columns[11]: "{:.1f}",
            columns[15]: "{:.1f}",
            columns[19]: "{:.1f}",
        },
    )
    styler.to_latex(
        "bin/table-uci-epilepsy-supervised.tex",
        hrules=True,
        multicol_align="c",
    )
    dataset_names = ["MNIST", "FashionMNIST"]
    dataset_list = [MNIST, FashionMNIST]
    accuracies_mnist_fashionmnist_supervised = []
    for dataset_name_index, (
        dataset_name,
        dataset,
        kernel_size_mnist_fashionmnist_range,
        mnist_fashionmnist_train_range,
        mnist_fashionmnist_validation_range,
        mnist_fashionmnist_test_range,
    ) in enumerate(
        zip(
            dataset_names,
            dataset_list,
            kernel_size_mnist_fashionmnist_ranges,
            mnist_fashionmnist_train_ranges,
            mnist_fashionmnist_validation_ranges,
            mnist_fashionmnist_test_ranges,
            strict=True,
        ),
    ):
        batch_size = 64
        lr = 0.01
        dataset_train_validation = dataset(
            "data",
            download=True,
            train=True,
            transform=ToTensor(),
        )
        dataloader_train = DataLoader(
            dataset_train_validation,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(mnist_fashionmnist_train_range),
        )
        dataloader_validation = DataLoader(
            dataset_train_validation,
            sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range),
            batch_size=batch_size,
        )
        dataset_test = dataset("data", train=False, transform=ToTensor())
        dataloader_test = DataLoader(
            dataset_test,
            sampler=SubsetRandomSampler(mnist_fashionmnist_test_range),
        )
        accuracy_best = 0
        fnn_model_supervised = FNN(
            len(dataset_train_validation.classes),
            dataset_train_validation.data[0],
        ).to(device)
        optimizer = optim.Adam(fnn_model_supervised.parameters(), lr=lr)
        for _ in range(epochs_num):
            fnn_model_supervised.train()
            for data, target in dataloader_train:
                data = data.to(device)  # noqa: PLW2901
                target = target.to(device)  # noqa: PLW2901
                output = fnn_model_supervised(data)
                classification_loss = functional.cross_entropy(output, target)
                optimizer.zero_grad()
                classification_loss.backward()  # type: ignore[no-untyped-call]
                optimizer.step()
            predictions_correct_num = 0
            predictions_num = 0
            fnn_model_supervised.eval()
            with torch.no_grad():
                for data, target in dataloader_validation:
                    data = data.to(device)  # noqa: PLW2901
                    target = target.to(device)  # noqa: PLW2901
                    output = fnn_model_supervised(data)
                    prediction = output.argmax(dim=1)
                    predictions_correct_num += sum(prediction == target).item()
                    predictions_num += output.shape[0]
            accuracy = 100 * predictions_correct_num / predictions_num
            if accuracy_best < accuracy:
                fnn_model_supervised_best = fnn_model_supervised
                accuracy_best = accuracy
        fnn_model_supervised.eval()
        with torch.no_grad():
            for data, target in dataloader_test:
                data = data.to(device)  # noqa: PLW2901
                target = target.to(device)  # noqa: PLW2901
                output = fnn_model_supervised_best(data)
                prediction = output.argmax(dim=1)
                predictions_correct_num += sum(prediction == target).item()
                predictions_num += output.shape[0]
        accuracies_mnist_fashionmnist_supervised.append(
            100 * predictions_correct_num / predictions_num,
        )
        sparse_activations = [
            Identity2D,
            Relu2D,
            TopKAbsolutes2D,
            ExtremaPoolIndices2D,
            Extrema2D,
        ]
        kernel_sizes_list = [
            2 * [kernel_size_mnist_fashionmnist]
            for kernel_size_mnist_fashionmnist in kernel_size_mnist_fashionmnist_range
        ]
        batch_size = 64
        lr = 0.01
        results_supervised_rows_list = []
        dataset_train_validation = dataset(
            "data",
            download=True,
            train=True,
            transform=ToTensor(),
        )
        dataloader_train = DataLoader(
            dataset_train_validation,
            batch_size=batch_size,
            sampler=SubsetRandomSampler(mnist_fashionmnist_train_range),
        )
        dataloader_validation = DataLoader(
            dataset_train_validation,
            sampler=SubsetRandomSampler(mnist_fashionmnist_validation_range),
        )
        dataset_test = dataset("data", train=False, transform=ToTensor())
        dataloader_test = DataLoader(
            dataset_test,
            sampler=SubsetRandomSampler(mnist_fashionmnist_test_range),
        )
        for kernel_sizes in kernel_sizes_list:
            results_supervised_rows = []
            for sparse_activation, sparse_activation_name in zip(
                sparse_activations,
                sparse_activation_names,
                strict=True,
            ):
                if sparse_activation == TopKAbsolutes2D:
                    sparsity_densities = [
                        int(dataset_test.data.shape[-1] / kernel_size) ** 2
                        for kernel_size in kernel_sizes
                    ]
                elif sparse_activation == Extrema2D:
                    sparsity_densities = np.clip(
                        [kernel_size - 2 for kernel_size in kernel_sizes],
                        1,
                        999,
                    ).tolist()
                    sparsity_densities = [[sparsity_density, sparsity_density] for sparsity_density in sparsity_densities]  # type: ignore[misc] # noqa: E501
                else:
                    sparsity_densities = kernel_sizes
                sparse_activation_list = [
                    sparse_activation(sparsity_density)
                    for sparsity_density in sparsity_densities
                ]
                san2d_model = SAN2d(kernel_sizes, sparse_activation_list).to(device)
                optimizer = optim.Adam(san2d_model.parameters(), lr=lr)
                hook_handles = [
                    Hook(sparse_activation_)
                    for sparse_activation_ in san2d_model.sparse_activations
                ]
                flithos_epoch_mean_best = float("inf")
                for _ in range(epochs_num):
                    train_model_unsupervised(dataloader_train, san2d_model, optimizer)
                    flithos_epoch, *_ = validate_or_test_model_unsupervised(
                        dataloader_validation,
                        hook_handles,
                        san2d_model,
                    )
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        san2d_model_epoch_best = san2d_model
                        flithos_epoch_mean_best = flithos_epoch.mean()
                for weights_kernel in san2d_model.weights_kernels:
                    weights_kernel.requires_grad_(False)  # noqa: FBT003
                flithos_epoch_mean_best = float("inf")
                fnn_model_supervised = FNN(
                    len(dataset_train_validation.classes),
                    dataset_train_validation.data[0],
                ).to(device)
                optimizer = optim.Adam(fnn_model_supervised.parameters(), lr=lr)
                for _ in range(epochs_num):
                    train_model_supervised(
                        dataloader_train,
                        fnn_model_supervised,
                        san2d_model_epoch_best,
                        optimizer,
                    )
                    flithos_epoch, *_ = validate_or_test_model_unsupervised(
                        dataloader_validation,
                        hook_handles,
                        san2d_model_epoch_best,
                    )
                    if flithos_epoch.mean() < flithos_epoch_mean_best:
                        fnn_model_supervised_best = fnn_model_supervised
                        san2d_model_best = san2d_model_epoch_best
                        flithos_epoch_mean_best = flithos_epoch.mean()
                (
                    flithos,
                    inverse_compression_ratio,
                    reconstruction_loss,
                    accuracy,
                ) = validate_or_test_model_supervised(
                    dataloader_test,
                    hook_handles,
                    fnn_model_supervised_best,
                    san2d_model_best,
                )
                results_supervised_rows.extend(
                    [
                        inverse_compression_ratio.mean(),
                        reconstruction_loss.mean(),
                        flithos.mean(),
                        accuracy
                        - accuracies_mnist_fashionmnist_supervised[dataset_name_index],
                    ],
                )
                if kernel_sizes[0] == 4:  # noqa: PLR2004
                    save_images_2d(
                        dataset_test[0][0][0],
                        dataset_name,
                        san2d_model_best,
                        sparse_activation_name.lower().replace(" ", "-"),
                    )
            results_supervised_rows_list.append(results_supervised_rows)
        header = [
            "$CR^{-1}$",
            "$\\tilde{\\mathcal{L}}$",
            "$\\bar\\varphi$",
            "A\\textsubscript{$\\pm$\\%}",
        ]
        columns = pd.MultiIndex.from_product([sparse_activation_names, header])
        results_supervised_df = pd.DataFrame(
            results_supervised_rows_list,
            columns=columns,
            index=kernel_size_mnist_fashionmnist_range,
        )
        results_supervised_df.index.names = ["$m$"]
        styler = results_supervised_df.style
        styler.format(
            precision=2,
            formatter={
                columns[3]: "{:.1f}",
                columns[7]: "{:.1f}",
                columns[11]: "{:.1f}",
                columns[15]: "{:.1f}",
                columns[19]: "{:.1f}",
            },
        )
        styler.to_latex(
            f"bin/table-{dataset_name.lower()}-supervised.tex",
            hrules=True,
            multicol_align="c",
        )
    keys_values_df = pd.DataFrame(
        {
            "key": [
                "uci-epilepsy-supervised-accuracy",
                "mnist-supervised-accuracy",
                "fashionmnist-supervised-accuracy",
            ],
            "value": [
                accuracy_uci_epilepsy,
                accuracies_mnist_fashionmnist_supervised[0],
                accuracies_mnist_fashionmnist_supervised[1],
            ],
        },
    )
    keys_values_df.to_csv("bin/keys-values.csv", index=False, float_format="%.2f")


if __name__ == "__main__":
    main()
