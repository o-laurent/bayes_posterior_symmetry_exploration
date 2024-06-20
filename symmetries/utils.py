import torch


def row_wise_multiplication(weight: torch.Tensor, lambdas: torch.Tensor):
    if weight.shape[0] != lambdas.shape[0]:
        raise ValueError(f"weight.shape[0] = {weight.shape[0]}, lambdas.shape = {lambdas.shape[0]}")
    weight *= lambdas[:, None]
    return weight


def col_wise_multiplication(weight: torch.Tensor, lambdas: torch.Tensor):
    if weight.shape[1] != lambdas.shape[0]:
        raise ValueError(f"weight.shape[1] = {weight.shape[1]}, lambdas.shape = {lambdas.shape[0]}")
    weight *= lambdas[None, :]
    return weight


def model_diff(model_1, model_2):
    diff = 0
    for param_1, param_2 in zip(model_1.parameters(), model_2.parameters(), strict=False):
        diff += torch.sum(torch.abs(param_1 - param_2))
    return diff
