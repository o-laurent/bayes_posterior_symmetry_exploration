"""Extracted and modified from https://github.com/antoninschrab/mmdagg-paper.

Originally MIT-Licenced by Antonin Schrab.
"""

from typing import Literal

import numpy as np
import torch


def mmdagg(
    X: np.ndarray,
    Y: np.ndarray,
    alpha: float = 0.05,
    kernel: str = "laplace_gaussian",
    number_bandwidths: int = 10,
    weights_type: Literal["uniform", "decreasing", "increasing", "centered"] = "uniform",
    B1: int = 2000,
    B2: int = 2000,
    B3: int = 50,
    seed: int = 42424242,
):
    # Assertions
    m = X.shape[0]
    n = Y.shape[0]
    assert n >= 2 and m >= 2
    assert alpha > 0 and alpha < 1
    assert kernel in (
        "gaussian",
        "laplace",
        "imq",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
        "all_matern_l1",
        "all_matern_l2",
        "all_matern_l1_l2",
        "all",
        "laplace_gaussian",
        "gaussian_laplace",
    )
    assert number_bandwidths > 1 and isinstance(number_bandwidths,int)
    assert weights_type in ("uniform", "decreasing", "increasing", "centered")
    assert B1 > 0 and isinstance(B1, int)
    assert B2 > 0 and isinstance(B2, int)
    assert B3 > 0 and isinstance(B3, int)

    approx_type = "permutations"

    # Collection of bandwidths
    # lambda_min / 2 * C^r for r = 0, ..., number_bandwidths -1
    # where C is such that lambda_max * 2 = lambda_min / 2 * C^(number_bandwidths - 1)
    def compute_bandwidths(distances, number_bandwidths):
        if np.min(distances) < 10 ** (-1):
            d = np.sort(distances)
            lambda_min = np.maximum(d[int(np.floor(len(d) * 0.05))], 10 ** (-1))
        else:
            lambda_min = np.min(distances)
        lambda_min = lambda_min / 2
        lambda_max = np.maximum(np.max(distances), 3 * 10 ** (-1))
        lambda_max = lambda_max * 2
        power = (lambda_max / lambda_min) ** (1 / (number_bandwidths - 1))
        return np.array([power**i * lambda_min for i in range(number_bandwidths)])

    max_samples = 500
    # bandwidths L1 for laplace, matern_0.5_l1, matern_1.5_l1, matern_2.5_l1, matern_3.5_l1, matern_4.5_l1

    distances_l1 = compute_l1_two_distance(X[:max_samples], Y[:max_samples]).reshape(-1)
    bandwidths_l1 = compute_bandwidths(distances_l1, number_bandwidths)
    # bandwidths L2 for gaussian, imq, matern_0.5_l2, matern_1.5_l2, matern_2.5_l2, matern_3.5_l2, matern_4.5_l2

    distances_l2 = compute_l2_two_distance(X[:max_samples], Y[:max_samples]).reshape(-1)
    bandwidths_l2 = compute_bandwidths(distances_l2, number_bandwidths)

    # Kernel and bandwidths list (order: "l1" first, "l2" second)
    if kernel in (
        "laplace",
        "matern_0.5_l1",
        "matern_1.5_l1",
        "matern_2.5_l1",
        "matern_3.5_l1",
        "matern_4.5_l1",
    ):
        kernel_bandwidths_l_list = [
            (kernel, bandwidths_l1, "l1"),
        ]
    elif kernel in (
        "gaussian",
        "imq",
        "matern_0.5_l2",
        "matern_1.5_l2",
        "matern_2.5_l2",
        "matern_3.5_l2",
        "matern_4.5_l2",
    ):
        kernel_bandwidths_l_list = [
            (kernel, bandwidths_l2, "l2"),
        ]
    elif kernel in ("laplace_gaussian", "gaussian_laplace"):
        kernel_bandwidths_l_list = [
            ("laplace", bandwidths_l1, "l1"),
            ("gaussian", bandwidths_l2, "l2"),
        ]
    elif kernel == "all_matern_l1":
        kernel_list = ["matern_" + str(i) + ".5_l1" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l1, "l1") for kernel in kernel_list]
    elif kernel == "all_matern_l2":
        kernel_list = ["matern_" + str(i) + ".5_l2" for i in range(5)]
        kernel_bandwidths_l_list = [(kernel, bandwidths_l2, "l2") for kernel in kernel_list]
    elif kernel == "all_matern_l1_l2":
        kernel_list = ["matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5)]
        bandwidths_list = [
            bandwidths_l1,
        ] * 5 + [
            bandwidths_l2,
        ] * 5
        l_list = [
            "l1",
        ] * 5 + [
            "l2",
        ] * 5
        kernel_bandwidths_l_list = [(kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(10)]
    elif kernel == "all":
        kernel_list = ["matern_" + str(i) + ".5_l" + str(j) for j in (1, 2) for i in range(5)] + ["gaussian", "imq"]
        bandwidths_list = (
            []
            + [
                bandwidths_l1,
            ]
            * 5
            + [
                bandwidths_l2,
            ]
            * 7
        )
        l_list = [
            "l1",
        ] * 5 + [
            "l2",
        ] * 7
        kernel_bandwidths_l_list = [(kernel_list[i], bandwidths_list[i], l_list[i]) for i in range(12)]
    else:
        raise ValueError("Kernel not defined.")

    # # Weights
    rs = np.random.RandomState(seed)
    if approx_type == "wild bootstrap":
        R = rs.choice([-1.0, 1.0], size=(B1 + B2 + 1, n))
        R[B1] = np.ones(n)
        R = R.transpose()
        R = np.concatenate((R, -R))  # (2n, B1+B2+1)
    elif approx_type == "permutations":
        idx = rs.rand(B1 + B2 + 1, m + n).argsort(axis=1)  # (B1+B2+1, m+n): rows of permuted indices
        # 11
        v11 = np.concatenate((np.ones(m), -np.ones(n)))  # (m+n, )
        V11i = np.tile(v11, (B1 + B2 + 1, 1))  # (B1+B2+1, m+n)
        V11 = np.take_along_axis(V11i, idx, axis=1)  # (B1+B2+1, m+n): permute the entries of the rows
        V11[B1] = v11  # (B1+1)th entry is the original MMD (no permutation)
        V11 = V11.transpose()  # (m+n, B1+B2+1)
        # 10
        v10 = np.concatenate((np.ones(m), np.zeros(n)))
        V10i = np.tile(v10, (B1 + B2 + 1, 1))
        V10 = np.take_along_axis(V10i, idx, axis=1)
        V10[B1] = v10
        V10 = V10.transpose()
        # 01
        v01 = np.concatenate((np.zeros(m), -np.ones(n)))
        V01i = np.tile(v01, (B1 + B2 + 1, 1))
        V01 = np.take_along_axis(V01i, idx, axis=1)
        V01[B1] = v01
        V01 = V01.transpose()
    else:
        raise ValueError("Approximation type not defined.")

    N = number_bandwidths * len(kernel_bandwidths_l_list)
    M = np.zeros((N, B1 + B2 + 1))
    last_l_pairwise_matrix_computed = ""
    for j in range(len(kernel_bandwidths_l_list)):
        kernel, bandwidths, l = kernel_bandwidths_l_list[j]
        # since kernel_bandwidths_l_list is ordered "l1" first, "l2" second
        # compute pairwise matrices the minimum amount of time
        # store only one pairwise matrix at once
        if l != last_l_pairwise_matrix_computed:
            pairwise_matrix = compute_pairwise_matrix(X, Y, l)
            last_l_pairwise_matrix_computed = l
        for i in range(number_bandwidths):
            bandwidth = bandwidths[i]
            K = kernel_matrix(pairwise_matrix, l, kernel, bandwidth)
            if approx_type == "wild bootstrap":
                # set diagonal elements of all four submatrices to zero
                np.fill_diagonal(K, 0)
                np.fill_diagonal(K[:n, n:], 0)
                np.fill_diagonal(K[n:, :n], 0)
                # compute MMD bootstrapped values
                M[number_bandwidths * j + i] = np.sum(R * (K @ R), 0)
            elif approx_type == "permutations":
                # set diagonal elements to zero
                np.fill_diagonal(K, 0)
                # compute MMD permuted values
                M[number_bandwidths * j + i] = (
                    np.sum(V10 * (K @ V10), 0) * (n - m + 1) / (m * n * (m - 1))
                    + np.sum(V01 * (K @ V01), 0) * (m - n + 1) / (m * n * (n - 1))
                    + np.sum(V11 * (K @ V11), 0) / (m * n)
                )
            else:
                raise ValueError("Approximation type not defined.")
    return M[:, B1]


def compute_l1_two_distance(v1, v2):
    return torch.cdist(
        torch.as_tensor(v1, dtype=torch.float32),
        torch.as_tensor(v2, dtype=torch.float32),
        p=1,
    ).numpy()


def compute_l2_two_distance(v1, v2):
    return torch.cdist(
        torch.as_tensor(v1, dtype=torch.float32),
        torch.as_tensor(v2, dtype=torch.float32),
        p=2,
    ).numpy()


def compute_l1_distance(v):
    t = torch.as_tensor(v, dtype=torch.float32)
    return torch.cdist(t, t, p=1).numpy()


def compute_l2_distance(v):
    t = torch.as_tensor(v, dtype=torch.float32)
    return torch.cdist(t, t, p=2).numpy()


def compute_pairwise_matrix(X, Y, l) -> np.ndarray:
    """Compute the pairwise distance matrix between all the points in X and Y,
    in L1 norm or L2 norm.

    inputs: X: (m,d) array of samples
            Y: (m,d) array of samples
            l: "l1" or "l2" or "l2sq"
    output: (2m,2m) pairwise distance matrix
    """
    Z = np.concatenate((X, Y))
    if l == "l1":
        return compute_l1_distance(Z)
    if l == "l2":
        return compute_l2_distance(Z)
    raise ValueError("Third input should either be 'l1' or 'l2'.")


def kernel_matrix(pairwise_matrix, l, kernel_type, bandwidth) -> np.ndarray:
    """Compute kernel matrix for a given kernel_type and bandwidth.

    inputs: pairwise_matrix: (2m,2m) matrix of pairwise distances
            l: "l1" or "l2" or "l2sq"
            kernel_type: string from ("gaussian", "laplace", "imq", "matern_0.5_l1", "matern_1.5_l1", "matern_2.5_l1", "matern_3.5_l1", "matern_4.5_l1", "matern_0.5_l2", "matern_1.5_l2", "matern_2.5_l2", "matern_3.5_l2", "matern_4.5_l2")
    output: (2m,2m) pairwise distance matrix

    Warning: The pair of variables l and kernel_type must be valid.
    """
    d = pairwise_matrix / bandwidth
    if kernel_type == "gaussian" and l == "l2":
        return np.exp(-(d**2))
    if kernel_type == "imq" and l == "l2":
        return (1 + d**2) ** (-0.5)
    if (kernel_type == "matern_0.5_l1" and l == "l1") or (kernel_type == "matern_0.5_l2" and l == "l2") or (kernel_type == "laplace" and l == "l1"):
        return np.exp(-d)
    if (kernel_type == "matern_1.5_l1" and l == "l1") or (kernel_type == "matern_1.5_l2" and l == "l2"):
        return (1 + np.sqrt(3) * d) * np.exp(-np.sqrt(3) * d)
    if (kernel_type == "matern_2.5_l1" and l == "l1") or (kernel_type == "matern_2.5_l2" and l == "l2"):
        return (1 + np.sqrt(5) * d + 5 / 3 * d**2) * np.exp(-np.sqrt(5) * d)
    if (kernel_type == "matern_3.5_l1" and l == "l1") or (kernel_type == "matern_3.5_l2" and l == "l2"):
        return (1 + np.sqrt(7) * d + 2 * 7 / 5 * d**2 + 7 * np.sqrt(7) / 3 / 5 * d**3) * np.exp(-np.sqrt(7) * d)
    if (kernel_type == "matern_4.5_l1" and l == "l1") or (kernel_type == "matern_4.5_l2" and l == "l2"):
        return (1 + 3 * d + 3 * (6**2) / 28 * d**2 + (6**3) / 84 * d**3 + (6**4) / 1680 * d**4) * np.exp(-3 * d)
    raise ValueError("The values of l and kernel_type are not valid.")


def create_weights(N, weights_type):
    """Create weights as defined in Section 5.1 of our paper.
    inputs: N: number of bandwidths to test
            weights_type: "uniform" or "decreasing" or "increasing" or "centered"
    output: (N,) array of weights.
    """
    if weights_type == "uniform":
        weights = np.array(
            [
                1 / N,
            ]
            * N
        )
    elif weights_type == "decreasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / (i * normaliser) for i in range(1, N + 1)])
    elif weights_type == "increasing":
        normaliser = sum([1 / i for i in range(1, N + 1)])
        weights = np.array([1 / ((N + 1 - i) * normaliser) for i in range(1, N + 1)])
    elif weights_type == "centered":
        if N % 2 == 1:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 1) for i in range(1, N + 1)])
            weights = np.array([1 / ((abs((N + 1) / 2 - i) + 1) * normaliser) for i in range(1, N + 1)])
        else:
            normaliser = sum([1 / (abs((N + 1) / 2 - i) + 0.5) for i in range(1, N + 1)])
            weights = np.array([1 / ((abs((N + 1) / 2 - i) + 0.5) * normaliser) for i in range(1, N + 1)])
    else:
        raise ValueError('The value of weights_type should be "uniform" or' '"decreasing" or "increasing" or "centered".')
    return weights
