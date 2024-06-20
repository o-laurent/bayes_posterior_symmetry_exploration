"""
This file contains a minimum working example of the scaling class
for fully connected networks. We will release the full code after
the anonymity period.
"""

import copy

import torch
from utils import col_wise_multiplication, row_wise_multiplication


def dimension(model: torch.nn.Module) -> list:
    """Compute the dimension of the scaling degrees of freedom of the model.

    Args:
        model (torch.nn.Module): The model to scale.

    Returns:
        List: The dimension of the scaling degrees of freedom of the model.
    """
    dim = [None]
    for _key, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.Linear):
            dim.append(mod.out_features)
        else:
            pass

    dim.pop(-1)
    dim.append(None)
    return dim


def _get_weights(model, squared=True):
    weights = []
    for _key, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.Linear):
            weights.append(mod.weight.detach() ** (2 if squared else 1))
        else:
            pass
    return weights


def _check_consistency(model, scaled_weights, verbose=False):
    inputs = torch.randn(16, 2)
    model.eval()
    model_hash = model(inputs).detach()
    scaled_model = copy.deepcopy(model)
    scaled_model.eval()
    scaled_model.load_state_dict(scaled_weights)
    scaled_hash = scaled_model(inputs)
    mse = torch.sum((scaled_hash - model_hash) ** 2).item()
    if mse > 1e-4:
        scaled_weights = None
    ending_mass = sum([torch.sum(w) for w in _get_weights(scaled_model)]) if scaled_weights is not None else None

    if verbose:
        print(f"Model MSE:{mse}")
        if scaled_weights is not None:
            print(f"Ending L2 mass {ending_mass}")
        else:
            print("Model integrity not preserved")
    return ending_mass


def _scale_linear(
    module: torch.nn.modules.Linear,
    lambda_mult: torch.Tensor = None,
    lambda_div: torch.Tensor = None,
) -> list[torch.Tensor]:
    weight = torch.clone(module.weight.detach())

    if lambda_div is not None:
        weight = col_wise_multiplication(weight, 1 / lambda_div)

    if lambda_mult is not None:
        weight = row_wise_multiplication(weight, lambda_mult)

    if module.bias is not None:
        bias = torch.clone(module.bias.detach())
        bias *= lambda_mult if lambda_mult is not None else 1
        weight = [weight, bias]
    else:
        weight = [weight]

    return weight


@torch.no_grad()
def scale(model, verbose=False):
    custom_scale = 3.0  # as defined in the paper to improve visuals
    model_dim = dimension(model)

    if verbose:
        starting_mass = sum([torch.sum(w) for w in _get_weights(model)])
        print(f"Starting L2 mass {starting_mass}")

    lambdas = [None]
    scaled_weights = {}
    layer_id = 0
    lambda_mult = None
    for key, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.Linear):
            if model_dim[layer_id + 1] is not None:
                if lambdas[-1] is not None:
                    lambda_mult = 1 / torch.norm(
                        col_wise_multiplication(mod.weight.detach().clone(), 1 / lambdas[-1]),
                        p=2,
                        dim=1,
                    )
                else:
                    lambda_mult = 1 / torch.norm(mod.weight.detach(), p=2, dim=1)

            else:
                lambda_mult = None

            if lambda_mult is not None:
                lambda_mult *= custom_scale

            cweight = _scale_linear(mod, lambda_mult=lambda_mult, lambda_div=lambdas[-1])
            lambdas.append(lambda_mult)

            scaled_weights[key + ".weight"] = cweight[0]
            if len(cweight) > 1:
                scaled_weights[key + ".bias"] = cweight[1]

            layer_id += 1
        else:
            pass

    _check_consistency(model, scaled_weights, verbose=verbose)

    return scaled_weights
