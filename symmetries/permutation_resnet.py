import copy

import torch
from torch_uncertainty.layers.filter_response_norm import FilterResponseNorm2d

from .layers import Residual


# TODO: handle flattens
class Permuter:
    unnaffected_layers = (
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.dropout.Dropout2d,
        torch.nn.modules.Dropout,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.linear.Identity,
        torch.nn.modules.batchnorm.BatchNorm2d,
        FilterResponseNorm2d,
        torch.nn.modules.Sequential,
    )

    def __init__(
        self,
        input_shape: tuple[int, ...],
        batch_size: int = 1,
        check_integrity: bool = True,
        verbose: bool = True,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.inputs = torch.randn(batch_size, *input_shape, dtype=dtype, device=device)
        self.check_integrity = check_integrity
        self.verbose = verbose

    def perm_dimension(self, model: torch.nn.Module) -> list[int]:
        dims = []
        for _, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                out_channels = mod.out_channels
                if mod.groups != 1:
                    dims.append([out_channels // mod.groups] * mod.groups)
                else:
                    dims.append(out_channels)

            elif isinstance(mod, torch.nn.modules.Linear):
                out_channels = mod.out_features
                dims.append(out_channels)

            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                pass

            elif isinstance(mod, Residual):
                dims.pop(-1)

            elif isinstance(mod, torch.nn.modules.flatten.Flatten | self.unnaffected_layers):
                pass

            else:
                print(f"Unknown layer type: {type(mod)}")

        dims.pop()

        return dims

    @torch.no_grad()
    def permute(self, model: torch.nn.Module, method: str = "max", model_type: str = "optunet") -> torch.Tensor:
        full_model = copy.deepcopy(model)
        for m in full_model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m.track_running_stats = False

        perm_weights = {}
        perm_left = None
        pis = []
        previous_key = None
        previous_mod = None
        for key, mod in model.named_modules():
            if model_type == "resnet" and key == "conv1":
                perm_weights[key + ".weight"] = torch.clone(mod.weight)
                perm_left = torch.arange(mod.out_channels)
            elif (
                model_type == "resnet"
                and "shortcut" in key
                and isinstance(
                    mod,
                    torch.nn.modules.Conv2d | torch.nn.modules.batchnorm.BatchNorm2d,
                )
            ):
                perm_weights[key + ".weight"] = mod.weight

                if mod.bias is not None:
                    perm_weights[key + ".bias"] = mod.bias

                if isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                    perm_weights[key + ".running_mean"] = torch.clone(mod.running_mean)
                    perm_weights[key + ".running_var"] = torch.clone(mod.running_var)
                    perm_weights[key + ".num_batches_tracked"] = torch.clone(mod.num_batches_tracked)

            elif isinstance(mod, torch.nn.modules.Conv2d):
                weight = torch.clone(mod.weight)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_left is not None and perm_left.numel() != 1 and mod.groups != 1 and previous_mod is not None:
                    previous_out_channels_per_group = previous_mod.out_channels // mod.groups
                    # Remove previous permutation
                    perm_weights[previous_key + ".weight"] = perm_weights[previous_key + ".weight"][torch.argsort(perm_left), ...]

                    if previous_mod.bias is not None:
                        perm_weights[previous_key + ".bias"] = perm_weights[previous_key + ".bias"][torch.argsort(perm_left)]

                    # Fix previous permutation for consistency over groups
                    true_perm_left_group = torch.argsort(
                        perm_weights[previous_key + ".weight"][:previous_out_channels_per_group, ...],
                        dim=0,
                        descending=True,
                    )[..., 0, 0, 0]

                    true_perm_left = []
                    if method == "max":
                        true_perm_left = [true_perm_left_group.clone() + previous_out_channels_per_group * i for i in range(mod.groups)]
                    else:
                        raise NotImplementedError("Only max method is implemented for groups")

                    true_perm_left = torch.cat(true_perm_left)

                    # Apply fixed previous permutation
                    perm_weights[previous_key + ".weight"] = perm_weights[previous_key + ".weight"][true_perm_left, ...]
                    if previous_key + ".bias" in perm_weights:
                        perm_weights[previous_key + ".bias"] = perm_weights[previous_key + ".bias"][true_perm_left]
                    pis.pop(-1)
                    pis.append(true_perm_left)

                    # Apply permutation to current layer
                    weight = weight[:, true_perm_left_group, ...]
                elif perm_left is not None and perm_left.numel() != 1:
                    weight = weight[:, perm_left, ...]
                    # No update of the bias here

                if mod.groups == 1:
                    if method == "max":
                        perm_left = torch.argsort(weight, dim=0, descending=True)[..., 0, 0, 0]
                    elif method == "mean":
                        perm_left = torch.argsort(
                            torch.mean(weight.view(mod.out_channels, -1), dim=-1),
                            dim=0,
                            descending=True,
                        )
                    elif method == "median":
                        perm_left = torch.argsort(
                            torch.median(weight.view(mod.out_channels, -1), dim=-1).values,
                            dim=0,
                            descending=True,
                        )
                else:
                    perms_left = []
                    out_channels_per_group = mod.out_channels // mod.groups
                    for i in range(mod.groups):
                        if method == "max":
                            perms_left.append(
                                torch.argsort(
                                    weight[i * out_channels_per_group : (i + 1) * out_channels_per_group],
                                    dim=0,
                                    descending=True,
                                )[..., 0, 0, 0]
                                + out_channels_per_group * i
                            )
                        else:
                            raise NotImplementedError("Only max method is implemented for groups")

                    perm_left = torch.cat(perms_left)

                if perm_left.numel() != 1:
                    weight = weight[perm_left, ...]
                    if mod.bias is not None:
                        bias = bias[perm_left]

                perm_weights[key + ".weight"] = weight

                if mod.bias is not None:
                    bias = bias.view(mod.bias.shape)
                    perm_weights[key + ".bias"] = bias

                previous_key = key
                previous_mod = mod

                pis.append(perm_left)

            elif isinstance(mod, torch.nn.modules.Linear):
                weight = torch.clone(mod.weight)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_left is not None and perm_left.numel() != 1:
                    weight = weight[:, perm_left]

                if method == "max":
                    perm_left = torch.argsort(weight, dim=0, descending=True)[..., 0]
                elif method == "mean":
                    perm_left = torch.argsort(torch.mean(weight, dim=-1), dim=0, descending=True)
                elif method == "median":
                    perm_left = torch.argsort(torch.median(weight, dim=-1).values, dim=0, descending=True)

                if perm_left.numel() != 1:
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
            elif isinstance(mod, torch.nn.modules.flatten.Flatten):
                # Do something
                continue
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                if model_type == "resnet" and "bn2" in key:
                    perm_weights[key + ".weight"] = torch.clone(mod.weight)
                    if mod.bias is not None:
                        perm_weights[key + ".bias"] = torch.clone(mod.bias)

                    perm_weights[key + ".running_mean"] = torch.clone(mod.running_mean)
                    perm_weights[key + ".running_var"] = torch.clone(mod.running_var)
                    perm_weights[key + ".num_batches_tracked"] = torch.clone(mod.num_batches_tracked)
                else:
                    perm_weights[key + ".weight"] = torch.clone(mod.weight)[perm_left]
                    if mod.bias is not None:
                        perm_weights[key + ".bias"] = torch.clone(mod.bias)[perm_left]

                    perm_weights[key + ".running_mean"] = torch.clone(mod.running_mean)[perm_left]
                    perm_weights[key + ".running_var"] = torch.clone(mod.running_var)[perm_left]
                    perm_weights[key + ".num_batches_tracked"] = torch.clone(mod.num_batches_tracked)
            elif isinstance(mod, FilterResponseNorm2d):
                if model_type == "resnet" and "bn2" in key:
                    perm_weights[key + ".weight"] = torch.clone(mod.weight)
                    if mod.bias is not None:
                        perm_weights[key + ".bias"] = torch.clone(mod.bias)

                    perm_weights[key + ".tau"] = torch.clone(mod.tau)
                    perm_weights[key + ".gamma"] = torch.clone(mod.gamma)
                    perm_weights[key + ".beta"] = torch.clone(mod.beta)

                else:
                    perm_weights[key + ".weight"] = torch.clone(mod.weight)[perm_left]
                    if mod.bias is not None:
                        perm_weights[key + ".bias"] = torch.clone(mod.bias)[perm_left]

                    perm_weights[key + ".tau"] = torch.clone(mod.tau)[perm_left]
                    perm_weights[key + ".gamma"] = torch.clone(mod.gamma)[perm_left]
                    perm_weights[key + ".beta"] = torch.clone(mod.beta)[perm_left]

            elif isinstance(mod, Residual):
                perm_weights[previous_key + ".weight"] = perm_weights[previous_key + ".weight"][torch.argsort(perm_left), ...]
                if previous_mod.bias is not None:
                    perm_weights[previous_key + ".bias"] = perm_weights[previous_key + ".bias"][torch.argsort(perm_left)]
                pis.pop(-1)
                pis.append(None)
                perm_left = None
            elif isinstance(mod, self.unnaffected_layers):
                continue

        if perm_left is not None:
            perm_weights[previous_key + ".weight"] = perm_weights[previous_key + ".weight"][torch.argsort(perm_left), ...]
            if previous_mod.bias is not None:
                perm_weights[previous_key + ".bias"] = perm_weights[previous_key + ".bias"][torch.argsort(perm_left)]
        pis.pop(-1)

        if self.check_integrity:
            full_model.eval()
            full_model = self.change_bn_eps(full_model)
            model_hash = full_model(self.inputs).cpu()
            full_model.load_state_dict(perm_weights)
            full_model = full_model.to(self.inputs.device)
            full_model.eval()
            full_model = self.change_bn_eps(full_model)
            perm_hash = full_model(self.inputs).cpu()
            mse = torch.sum((perm_hash - model_hash) ** 2 / self.inputs.shape[0]).item()
            if mse > 1:
                raise ValueError(f"Model integrity check failed. MSE: {mse}.")
            if self.verbose:
                print(f"Model MSE:{mse}")

        return perm_weights, pis

    def permute_weights(self, model: torch.nn.Module, pis: list) -> torch.Tensor:
        dims = self.perm_dimension(model)
        if len(dims) != len(pis):
            raise ValueError(f"Permutation dimension must be correct. {len(dims)} != {len(pis)}")

        self.model = model
        self.model_hash = model(self.inputs).detach()

        perm_weights = {}
        layer_id = 0
        perm_left, perm_right = None, None
        for key in model._modules:
            mod = model._modules[key]
            if isinstance(mod, torch.nn.modules.Conv2d):
                out_channels = mod.out_channels // mod.groups
                perm_left = pis[layer_id] if layer_id < len(pis) else torch.arange(out_channels)

                weight = torch.clone(mod.weight)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_right is not None and perm_right.numel() != 1:
                    weight = weight[:, perm_right, ...]

                if perm_left.numel() != 1:
                    weight = weight[perm_left, ...]  # TODO change
                    if mod.bias is not None:
                        bias = bias[perm_left]

                perm_weights[key + ".weight"] = weight

                if mod.bias is not None:
                    bias = bias.view(mod.bias.shape)
                    perm_weights[key + ".bias"] = bias

                layer_id += 1

            elif isinstance(mod, torch.nn.modules.Linear):
                out_channels = mod.out_features
                weight = torch.clone(mod.weight)
                perm_left = pis[layer_id] if layer_id < len(pis) else torch.arange(out_channels)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_right is not None and perm_right.numel() != 1:
                    weight = weight[:, torch.argsort(perm_right)]

                if perm_left.numel() != 1:
                    weight = weight[perm_left, :]
                    if mod.bias is not None:
                        bias = bias[perm_left]

                perm_weights[key + ".weight"] = weight

                if mod.bias is not None:
                    bias = bias.view(mod.bias.shape)
                    perm_weights[key + ".bias"] = bias

                layer_id += 1

            elif isinstance(mod, torch.nn.modules.flatten.Flatten):
                # Do something
                pass
            elif isinstance(mod, self.unnaffected_layers):
                pass
            else:
                raise TypeError(f"Unknown layer type: {type(mod)}")
            if perm_left is not None:
                perm_right = torch.argsort(perm_left)

        # Remove permutations from the last layer works only if linear
        perm_weights[key + ".weight"] = perm_weights[key + ".weight"][perm_right, :]
        if mod.bias is not None:
            perm_weights[key + ".bias"] = perm_weights[key + ".bias"][perm_right]

        if self.check_integrity:
            perm_model = copy.deepcopy(self.model)
            perm_model.load_state_dict(perm_weights)
            perm_hash = perm_model(self.inputs)
            mse = torch.sum((perm_hash - self.model_hash) ** 2).item()
            if mse > 1e-6:
                raise ValueError(f"Model integrity check failed. MSE: {mse}.")
            if self.verbose:
                print(f"Model MSE:{mse}")

        return perm_weights

    def random(self, model: torch.nn.Module) -> torch.Tensor:
        self.model = model
        self.model_hash = model(self.inputs).detach()

        perm_weights = {}
        layer_id = 0
        pis = []
        perm_left, perm_right = None, None
        y = self.inputs
        for key in model._modules:
            mod = model._modules[key]
            if isinstance(mod, torch.nn.modules.Conv2d):
                out_channels = mod.out_channels // mod.groups
                if mod.groups != 1:
                    perm_left = []
                    for i in range(mod.groups):
                        perm_left.append(torch.randperm(out_channels) + out_channels * i)
                    perm_left = torch.cat(perm_left)
                else:
                    perm_left = torch.randperm(out_channels)

                weight = torch.clone(mod.weight)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_right is not None and perm_right.numel() != 1:
                    weight = weight[:, perm_right, ...]

                if perm_left.numel() != 1:
                    weight = weight[perm_left, ...]  # TODO change
                    if mod.bias is not None:
                        bias = bias[perm_left]

                pis.append(perm_left)
                perm_weights[key + ".weight"] = weight

                if mod.bias is not None:
                    bias = bias.view(mod.bias.shape)
                    perm_weights[key + ".bias"] = bias

                layer_id += 1
                y = mod(y)

            elif isinstance(mod, torch.nn.modules.Linear):
                out_channels = mod.out_features
                weight = torch.clone(mod.weight)
                perm_left = torch.randperm(out_channels)

                if mod.bias is not None:
                    bias = torch.clone(mod.bias)

                if perm_right is not None and perm_right.numel() != 1:
                    weight = weight[:, torch.argsort(perm_right)]

                if perm_left.numel() != 1:
                    weight = weight[perm_left, :]
                    if mod.bias is not None:
                        bias = bias[perm_left]

                pis.append(perm_left)
                perm_weights[key + ".weight"] = weight

                if mod.bias is not None:
                    bias = bias.view(mod.bias.shape)
                    perm_weights[key + ".bias"] = bias

                layer_id += 1

            elif isinstance(mod, torch.nn.modules.flatten.Flatten | self.unnaffected_layers):
                pass
            else:
                raise TypeError(f"Unknown layer type: {type(mod)}")
            if perm_left is not None:
                perm_right = torch.argsort(perm_left)
        pis.pop()

        # Remove permutations from the last layer
        perm_weights[key + ".weight"] = perm_weights[key + ".weight"][perm_right, :]
        if mod.bias is not None:
            perm_weights[key + ".bias"] = perm_weights[key + ".bias"][perm_right]

        if self.check_integrity:
            perm_model = copy.deepcopy(self.model)
            perm_model.load_state_dict(perm_weights)
            perm_hash = perm_model(self.inputs)
            mse = torch.sum((perm_hash - self.model_hash) ** 2).item()
            if mse > 1e-6:
                raise ValueError(f"Model integrity check failed. MSE: {mse}.")
            if self.verbose:
                print(f"Model MSE:{mse}")

        return perm_weights, pis

    def change_bn_eps(self, model, eps_value: float = 0.0):
        for _key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                mod.eps = eps_value
            else:
                pass
        return model
