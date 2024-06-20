import copy

import torch


@torch.no_grad()
def permute(model, custom_pis=None, verbose=False):
    inputs = torch.randn(16, 2)

    full_model = copy.deepcopy(model)

    perm_weights = {}
    pis = []
    perm_left, _perm_right = None, None
    previous_key, previous_mod = None, None
    layer = 0
    for key, mod in model.named_modules():
        if isinstance(mod, torch.nn.modules.Linear):
            weight = torch.clone(mod.weight)

            if mod.bias is not None:
                bias = torch.clone(mod.bias)

            if perm_left is not None and perm_left.numel() != 1:
                weight = weight[:, perm_left]

            perm_left = torch.argsort(weight, dim=0, descending=True)[..., 0] if custom_pis is None else custom_pis[layer]

            if perm_left is not None and perm_left.numel() != 1:
                weight = weight[perm_left, :]
                if mod.bias is not None:
                    bias = bias[perm_left]

            perm_weights[key + ".weight"] = weight

            if mod.bias is not None:
                bias = bias.view(mod.bias.shape)
                perm_weights[key + ".bias"] = bias
            previous_key = key
            previous_mod = mod

            pis.append(perm_left)
            layer += 1
        else:
            pass

    if perm_left is not None:
        perm_weights[previous_key + ".weight"] = perm_weights[previous_key + ".weight"][torch.argsort(perm_left), :]
        if previous_mod.bias is not None:
            perm_weights[previous_key + ".bias"] = perm_weights[previous_key + ".bias"][torch.argsort(perm_left)]
    pis.pop(-1)

    full_model.eval()
    model_hash = full_model(inputs).cpu()
    full_model.load_state_dict(perm_weights)
    full_model = full_model.to(inputs.device)
    full_model.eval()
    perm_hash = full_model(inputs).cpu()
    mse = torch.sum((perm_hash - model_hash) ** 2 / inputs.shape[0]).item()
    if verbose:
        if mse > 1e-4:
            raise ValueError(f"Model integrity check failed. MSE: {mse}.")
        print(f"Model MSE:{mse}")

    return perm_weights, pis
