"""Sparsely activated networks."""  # noqa: INP001

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


class Extrema1D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, minimum_extrema_distance: int) -> None:  # noqa: D107
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _extrema_1d(input_, self.minimum_extrema_distance)


class Extrema2D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, minimum_extrema_distance: list[int]) -> None:  # noqa: D107
        super().__init__()
        self.minimum_extrema_distance = minimum_extrema_distance

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _extrema_2d(input_, self.minimum_extrema_distance)


class ExtremaPoolIndices1D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, pool_size: int) -> None:  # noqa: D107
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _extrema_pool_indices_1d(input_, self.pool_size)


class ExtremaPoolIndices2D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, pool_size: int) -> None:  # noqa: D107
        super().__init__()
        self.pool_size = pool_size

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _extrema_pool_indices_2d(input_, self.pool_size)


class SAN1d(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(  # noqa: D107
        self: SAN1d,
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

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:  # noqa: D102
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


class SAN2d(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(  # noqa: D107
        self: SAN2d,
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

    def forward(self, batch_x: torch.Tensor) -> torch.Tensor:  # noqa: D102
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


class TopKAbsolutes1D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, topk: int) -> None:  # noqa: D107
        super().__init__()
        self.topk = topk

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _topk_absolutes_1d(input_, self.topk)


class TopKAbsolutes2D(nn.Module):  # type: ignore[misc] # noqa: D101
    def __init__(self, topk: int) -> None:  # noqa: D107
        super().__init__()
        self.topk = topk

    def forward(self, input_: torch.Tensor) -> torch.Tensor:  # noqa: D102
        return _topk_absolutes_2d(input_, self.topk)


def _extrema_1d(input_: torch.Tensor, minimum_extrema_distance: int) -> torch.Tensor:
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


def _extrema_2d(
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


def _extrema_pool_indices_1d(input_: torch.Tensor, kernel_size: int) -> torch.Tensor:
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


def _extrema_pool_indices_2d(input_: torch.Tensor, kernel_size: int) -> torch.Tensor:
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


def _topk_absolutes_1d(input_: torch.Tensor, topk: int) -> torch.Tensor:
    extrema_primary = torch.zeros_like(input_)
    _, extrema_indices = torch.topk(abs(input_), topk)
    return extrema_primary.scatter(
        -1,
        extrema_indices,
        input_.gather(-1, extrema_indices),
    )


def _topk_absolutes_2d(input_: torch.Tensor, topk: int) -> torch.Tensor:
    x_flattened = input_.view(input_.shape[0], -1)
    extrema_primary = torch.zeros_like(x_flattened)
    _, extrema_indices = torch.topk(abs(x_flattened), topk)
    return extrema_primary.scatter(
        -1,
        extrema_indices,
        x_flattened.gather(-1, extrema_indices),
    ).view(input_.shape)
