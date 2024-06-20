import numpy as np
import torch


def to_vector(model):
    shapes = {}
    models = []
    if isinstance(model, dict) or len(model) == 1:
        layers = []
        for key, val in model.items():
            layers.append(val.flatten().cpu())
            shapes[key.replace("model.", "")] = val.shape
        return torch.concatenate(layers), shapes
    for i in range(len(model)):
        layers = []
        for key, val in model[i].items():
            layers.append(val.flatten().cpu())
            shapes[key.replace("model.", "")] = val.shape
        models.append(torch.concatenate(layers))
    return torch.stack(models), shapes


def to_state_dict(model, shapes):
    state_dict = {}
    for key, shape in shapes.items():
        state_dict[key] = model[: np.prod(shape)].reshape(shape)
        model = model[np.prod(shape) :]
    return state_dict


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
