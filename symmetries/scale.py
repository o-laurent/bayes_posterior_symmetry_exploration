import copy

import cvxpy as cp
import numpy as np
import scipy.optimize
import torch
from torch_symmetry.layers import Residual
from torch_symmetry.utils import col_wise_multiplication, row_wise_multiplication


class Scaler:
    """Class to scale the weights of a model.

    Args:
        input_shape (tuple): The shape of the input of the model.
        optimize (bool, optional): Whether to optimize the scaling. Defaults to True.
        check_integrity (bool, optional): Whether to check the integrity of the model after
            scaling. Defaults to True.
        optim_algorithm (str, optional): The optimization algorithm to use. Defaults to
            "nelder-mead".
        optim_xatol (float, optional): The tolerance of the optimization algorithm. Defaults to
            1e-6.
        optim_maxiter (int, optional): The maximum number of iterations of the optimization
            algorithm. Defaults to 10000.
        verbose (bool, optional): Whether to print information about the scaling. Defaults to
            True.
    """

    unnaffected_layers = (
        torch.nn.modules.activation.ReLU,
        torch.nn.modules.dropout.Dropout2d,
        torch.nn.modules.Dropout,
        torch.nn.modules.pooling.MaxPool2d,
        torch.nn.modules.pooling.AdaptiveAvgPool2d,
        torch.nn.modules.linear.Identity,
        torch.nn.modules.container.Sequential,
    )
    integrity_tol = 1e-4

    def __init__(
        self,
        input_shape: tuple,
        batch_size: int = 1,
        optimize: bool = True,
        check_integrity: bool = True,
        verbose: bool = True,
        device: torch.device | None = None,
    ):
        if device is None:
            device = torch.device("cpu")
        self.inputs = torch.randn(batch_size, *input_shape, device=device)
        self.optimize = optimize
        self.check_integrity = check_integrity
        self.verbose = verbose

    def dimension(self, model: torch.nn.Module) -> list:
        """Compute the dimension of the scaling degrees of freedom of the model.

        Args:
            model (torch.nn.Module): The model to scale.

        Returns:
            List: The dimension of the scaling degrees of freedom of the model.
        """
        dim = [None]
        for _key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                dim.append(mod.out_channels)
            elif isinstance(mod, torch.nn.modules.Linear):
                dim.append(mod.out_features)
            elif isinstance(mod, torch.nn.modules.flatten.Flatten):
                # Do something
                continue
            elif isinstance(mod, Residual):
                dim.pop(-1)
                dim.append(None)
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                dim.append(mod.num_features)
            elif isinstance(mod, self.unnaffected_layers):
                continue
            else:
                pass

        dim.pop(-1)
        dim.append(None)
        return dim

    def dimension_resnet(self, model: torch.nn.Module) -> list:
        """Compute the dimension of the scaling degrees of freedom of the model.

        Args:
            model (torch.nn.Module): The model to scale.

        Returns:
            List: The dimension of the scaling degrees of freedom of the model.
        """
        dim = [None]
        for key, mod in model.named_modules():
            if "shortcut" in key:
                continue
            if isinstance(mod, torch.nn.modules.Conv2d):
                dim.append(mod.out_channels)
            elif isinstance(mod, torch.nn.modules.Linear):
                dim.append(mod.out_features)
            elif isinstance(mod, torch.nn.modules.flatten.Flatten):
                # Do something
                continue
            elif isinstance(mod, Residual):
                dim.pop(-1)
                dim.append(None)
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                dim.append(mod.num_features)
            elif isinstance(mod, self.unnaffected_layers):
                continue
            else:
                pass

        dim.pop(-1)
        dim.append(None)
        return dim

    def total_dimension(self, model):
        return sum(filter(None, self.dimension(model)))

    @torch.no_grad()
    def scale(
        self,
        model: torch.nn.Module,
        method: str,
        custom_scale: float = 1.0,
        custom_lambdas: list | None = None,
        algorithm: str = "nelder-mead",
    ) -> dict | float:
        model = copy.deepcopy(model)
        if method == "unitnorm" or method == "unitvar" or method == "manual":
            return self._scale_stats(
                model,
                method=method,
                custom_scale=custom_scale,
                custom_lambdas=custom_lambdas,
            )
        if method == "scipyopt":
            return self._scale_scipy_optimize(model, algorithm=algorithm)
        if method == "cvxopt":
            return self._scale_cvx_optimize(model)
        raise NotImplementedError(f"Method {method} not implemented")

    @torch.no_grad()  # prendre en compte les residual
    def _scale_stats(
        self,
        model,
        method: str = "unitnorm",
        custom_scale: float = 1.0,
        custom_lambdas: list | None = None,
    ):
        model_dim = self.dimension(model)
        if self.verbose:
            starting_mass = sum([torch.sum(w) for w in self._get_weights(model)])
            print(f"Starting L2 mass {starting_mass}")

        lambdas = [None]
        scaled_weights = {}
        layer_id = 0
        lambda_mult = None
        for key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                if model_dim[layer_id + 1] is not None:
                    if lambdas[-1] is not None:
                        if method == "unitvar":
                            lambda_mult = 1 / self._scale_conv(mod, lambda_mult=None, lambda_div=lambdas[-1])[0].view(mod.out_channels, -1).std(dim=-1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(
                                self._scale_conv(mod, lambda_mult=None, lambda_div=lambdas[-1])[0].view(mod.out_channels, -1),
                                p=2,
                                dim=-1,
                            )
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]
                    else:
                        if method == "unitvar":
                            lambda_mult = 1 / mod.weight.view(mod.out_channels, -1).std(dim=-1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(mod.weight.view(mod.out_channels, -1), p=2, dim=-1)
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]

                else:
                    lambda_mult = None

                if lambda_mult is not None:
                    lambda_mult *= custom_scale

                cweight = self._scale_conv(mod, lambda_mult=lambda_mult, lambda_div=lambdas[-1])
                lambdas.append(lambda_mult)

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

                layer_id += 1

            elif isinstance(mod, torch.nn.modules.Linear):
                if model_dim[layer_id + 1] is not None:
                    if lambdas[-1] is not None:
                        if method == "unitvar":
                            lambda_mult = 1 / col_wise_multiplication(mod.weight.detach().clone(), 1 / lambdas[-1]).std(dim=1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(
                                col_wise_multiplication(mod.weight.detach().clone(), 1 / lambdas[-1]),
                                p=2,
                                dim=1,
                            )
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]
                    else:
                        if method == "unitvar":
                            lambda_mult = 1 / mod.weight.detach().std(dim=1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(mod.weight.detach(), p=2, dim=1)
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]

                else:
                    lambda_mult = None

                if lambda_mult is not None:
                    lambda_mult *= custom_scale

                cweight = self._scale_linear(mod, lambda_mult=lambda_mult, lambda_div=lambdas[-1])
                lambdas.append(lambda_mult)

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

                layer_id += 1
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                if model_dim[layer_id + 1] is not None:
                    if method in ["unitvar", "unitnorm"]:
                        lambda_mult = (
                            1
                            / self._scale_bn(
                                copy.deepcopy(mod),
                                lambda_mult=None,
                                lambda_div=lambdas[-1],
                            )[3]
                        )
                    elif method == "manual":
                        lambda_mult = custom_lambdas[layer_id]

                (
                    running_mean,
                    running_var,
                    num_batches_tracked,
                    weight,
                    bias,
                ) = self._scale_bn(mod, lambda_mult=lambda_mult, lambda_div=lambdas[-1])
                lambdas.append(lambda_mult)
                scaled_weights[key + ".running_mean"] = running_mean.clone()
                scaled_weights[key + ".running_var"] = running_var.clone()
                scaled_weights[key + ".num_batches_tracked"] = num_batches_tracked.clone()

                scaled_weights[key + ".weight"] = weight.clone()
                if bias is not None:
                    scaled_weights[key + ".bias"] = bias.clone()
                layer_id += 1
            else:
                pass

        end_mass = self._check_consistency(model, scaled_weights)

        return scaled_weights, end_mass

    @torch.no_grad()  # prendre en compte les residual
    def _scale_stats_resnet(
        self,
        model,
        method: str = "unitnorm",
        custom_scale: float = 1.0,
        custom_lambdas=None,
    ):
        model_dim = self.dimension_resnet(model)
        if self.verbose:
            print(np.sum([torch.sum(w**2).item() for w in model.state_dict().values()]))

        lambdas = [None]
        shortcut_lambdas = []
        scaled_weights = {}
        layer_id = 0
        for key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                if model_dim[layer_id + 1] is not None or "shortcut" in key:
                    lambda_div = None if "shortcut" in key else lambdas[-1]
                    if lambda_div is not None:
                        if method == "unitvar":
                            lambda_mult = 1 / self._scale_conv(mod, lambda_mult=None, lambda_div=lambda_div)[0].view(mod.out_channels, -1).std(dim=-1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(
                                self._scale_conv(mod, lambda_mult=None, lambda_div=lambda_div)[0].view(mod.out_channels, -1),
                                p=2,
                                dim=-1,
                            )
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]
                    else:
                        if method == "unitvar":
                            lambda_mult = 1 / mod.weight.view(mod.out_channels, -1).std(dim=-1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(mod.weight.view(mod.out_channels, -1), p=2, dim=-1)
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]

                else:
                    lambda_mult = None

                if lambda_mult is not None:
                    lambda_mult *= custom_scale

                cweight = self._scale_conv(mod, lambda_mult=lambda_mult, lambda_div=lambda_div)
                if "shortcut" in key:
                    shortcut_lambdas.append(lambda_mult)
                else:
                    lambdas.append(lambda_mult)

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

                if "shortcut" not in key:
                    layer_id += 1

            elif isinstance(mod, torch.nn.modules.Linear):
                if model_dim[layer_id + 1] is not None:
                    if lambdas[-1] is not None:
                        if method == "unitvar":
                            lambda_mult = 1 / col_wise_multiplication(mod.weight.detach().clone(), 1 / lambdas[-1]).std(dim=1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(
                                col_wise_multiplication(mod.weight.detach().clone(), 1 / lambdas[-1]),
                                p=2,
                                dim=1,
                            )
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]
                    else:
                        if method == "unitvar":
                            lambda_mult = 1 / mod.weight.detach().std(dim=1)
                        elif method == "unitnorm":
                            lambda_mult = 1 / torch.norm(mod.weight.detach(), p=2, dim=1)
                        elif method == "manual":
                            lambda_mult = custom_lambdas[layer_id]
                else:
                    lambda_mult = None

                if lambda_mult is not None:
                    lambda_mult *= custom_scale

                cweight = self._scale_linear(mod, lambda_mult=lambda_mult, lambda_div=lambdas[-1])
                lambdas.append(lambda_mult)

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

                layer_id += 1
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                if model_dim[layer_id + 1] is not None:
                    if "shortcut" in key:
                        lambda_mult = None
                        lambda_div = shortcut_lambdas[-1]
                    else:
                        lambda_div = lambdas[-1]
                        lambda_mult = (
                            1
                            / self._scale_bn(
                                copy.deepcopy(mod),
                                lambda_mult=None,
                                lambda_div=lambda_div,
                            )[3]
                        )
                else:
                    if "shortcut" in key:
                        lambda_div = shortcut_lambdas[-1]
                        lambda_mult = None
                    else:
                        lambda_div = lambdas[-1]
                        lambda_mult = None

                (
                    running_mean,
                    running_var,
                    num_batches_tracked,
                    weight,
                    bias,
                ) = self._scale_bn(mod, lambda_mult=lambda_mult, lambda_div=lambda_div)
                if key == "bn1":
                    shortcut_lambdas.append(lambda_mult)
                    lambdas.append(lambda_mult)
                elif "shortcut" in key:
                    shortcut_lambdas.append(lambda_mult)
                else:
                    lambdas.append(lambda_mult)
                scaled_weights[key + ".running_mean"] = running_mean.clone()
                scaled_weights[key + ".running_var"] = running_var.clone()
                scaled_weights[key + ".num_batches_tracked"] = num_batches_tracked.clone()

                scaled_weights[key + ".weight"] = weight.clone()
                if bias is not None:
                    scaled_weights[key + ".bias"] = bias.clone()

                if "shortcut" not in key:
                    layer_id += 1
            else:
                pass

        self._check_consistency(model, scaled_weights)

        return scaled_weights

    def _check_consistency(self, model, scaled_weights):
        model.eval()
        model = self.change_bn_eps(model)
        model_hash = model(self.inputs).detach()
        scaled_model = copy.deepcopy(model)
        scaled_model.eval()
        scaled_model = self.change_bn_eps(scaled_model)
        scaled_model.load_state_dict(scaled_weights)
        scaled_hash = scaled_model(self.inputs)
        mse = torch.sum((scaled_hash - model_hash) ** 2).item()
        if mse > self.integrity_tol:
            scaled_weights = None
        ending_mass = sum([torch.sum(w) for w in self._get_weights(scaled_model)]) if scaled_weights is not None else None
        if self.verbose:
            print(f"Model MSE:{mse}")
            if scaled_weights is not None:
                print(f"Ending L2 mass {ending_mass}")
            else:
                print("Model integrity not preserved")
        return ending_mass

    def _scale_scipy_optimize(self, model, algorithm):
        self.model = copy.deepcopy(model)
        lambda_0 = torch.ones(self.total_dimension(model))
        res = scipy.optimize.minimize(
            self._eval_mass_log,
            lambda_0,
            method=algorithm,
        )
        opt_lam = torch.sqrt(torch.exp(torch.tensor(res.x)))
        opt_lam = self._split_lambdas(model, opt_lam)

        scaled_weights, end_mass = self._scale_stats(model.float(), method="manual", custom_lambdas=opt_lam[1:-1])
        return opt_lam, scaled_weights, end_mass

    def _scale_cvx_optimize(self, model, model_type, verbose=True):
        mass, lambdas = self._create_cvxpy_problem(model, model_type)
        prob = cp.Problem(cp.Minimize(mass))
        prob.solve(verbose=verbose)
        if prob.status != "optimal":
            raise RuntimeError("Optimization failed")
        opt_lam = [torch.exp(torch.tensor(lam.value)) if lam is not None else None for lam in lambdas]
        if model_type == "resnet":
            scaled_weights = self._scale_stats_resnet(model, method="manual", custom_lambdas=opt_lam[1:-1])
        else:
            scaled_weights, _ = self._scale_stats(model, method="manual", custom_lambdas=opt_lam[1:-1])
        return opt_lam, scaled_weights

    def _create_cvxpy_problem(self, model, model_type="resnet"):
        weights = self._get_weights(model, squared=True)
        if model_type == "resnet":
            dims = self.dimension_resnet(model)
            print(dims)
        else:
            dims = self.dimension(model)
        lambdas = []
        for dim in dims:
            if dim is None:
                lambdas.append(None)
            else:
                lambdas.append(cp.Variable(dim))

        mass = 0
        num_mods = len(weights)
        for i in range(1, num_mods + 1):
            mod_weight = weights[i - 1]
            if mod_weight.ndim == 1:  # BatchNorm
                if lambdas[i - 1] is not None and lambdas[i] is None:
                    mod_weight = cp.multiply(mod_weight, cp.exp(-2 * lambdas[i - 1]))
                elif lambdas[i] is not None and lambdas[i - 1] is None:
                    mod_weight = cp.multiply(cp.exp(2 * lambdas[i]), mod_weight)
                elif lambdas[i] is not None and lambdas[i - 1] is not None:
                    omega_scale = cp.exp(2 * (lambdas[i] - lambdas[i - 1]))
                    mod_weight = cp.multiply(mod_weight, omega_scale)

            if mod_weight.ndim == 2:  # Linear & summed Conv2D
                if lambdas[i - 1] is not None and lambdas[i] is None:
                    mod_weight = mod_weight @ cp.diag(cp.exp(-2 * lambdas[i - 1]))
                elif lambdas[i] is not None and lambdas[i - 1] is None:
                    mod_weight = cp.diag(cp.exp(2 * lambdas[i])) @ mod_weight
                elif lambdas[i] is not None and lambdas[i - 1] is not None:
                    tmp_div = np.ones((dims[i], 1)) @ cp.reshape(lambdas[i - 1], (1, dims[i - 1]))
                    tmp_mult = cp.reshape(lambdas[i], (dims[i], 1)) @ np.ones((1, dims[i - 1]))
                    omega_scale = cp.exp(2 * (tmp_mult - tmp_div))
                    mod_weight = cp.multiply(mod_weight, omega_scale)

            mass += cp.sum(mod_weight)

        return mass, lambdas

    def _eval_mass_log(self, lambdas):
        weights = self._get_weights(self.model, squared=True)
        lambdas = torch.as_tensor(lambdas, dtype=torch.double)
        split_lambdas = self._split_lambdas(self.model, lambdas)

        def _inner_eval_mass_log(inner_lambdas):
            num_mods = len(weights)
            dims = self.dimension(self.model)
            mass = 0
            masses = []
            for i in range(1, num_mods + 1):
                mod_weight = weights[i - 1].clone()
                if mod_weight.ndim == 1:  # BatchNorm
                    if inner_lambdas[i] is not None:
                        mod_weight = mod_weight * torch.exp(2 * inner_lambdas[i])

                    masses.append(torch.sum(mod_weight))
                    mass += torch.sum(mod_weight)

                elif mod_weight.ndim == 2:  # Linear
                    if inner_lambdas[i - 1] is not None and inner_lambdas[i] is None:
                        mod_weight = mod_weight @ torch.diag(torch.exp(-2 * inner_lambdas[i - 1]))
                    elif inner_lambdas[i] is not None and inner_lambdas[i - 1] is None:
                        mod_weight = torch.diag(torch.exp(2 * inner_lambdas[i])) @ mod_weight
                    elif inner_lambdas[i] is not None and inner_lambdas[i - 1] is not None:
                        tmp_div = torch.ones((dims[i], 1), dtype=torch.double) @ inner_lambdas[i - 1].reshape(1, dims[i - 1])

                        tmp_mult = inner_lambdas[i].reshape(dims[i], 1) @ torch.ones((1, dims[i - 1]), dtype=torch.double)
                        omega_scale = torch.exp(2 * (tmp_mult - tmp_div))
                        mod_weight = mod_weight * omega_scale

                    masses.append(torch.sum(mod_weight))
                    mass += torch.sum(mod_weight)
            return mass

        return _inner_eval_mass_log(split_lambdas)

    def _split_lambdas(self, model, lambdas):
        dims = self.dimension(model)
        split_lambdas = []
        for i in range(len(dims)):
            if dims[i] is None:
                split_lambdas.append(None)
            else:
                size = dims[i]
                split_lambdas.append(lambdas[:size])
                lambdas = lambdas[size:]
        return split_lambdas

    def _eval_mass(self, weights, lambdas):
        num_mods = len(weights)
        mass = 0
        num_mods = len(weights)
        mass = 0
        for i in range(1, num_mods + 1):
            mod_weight = weights[i - 1]
            if mod_weight.ndim == 1:  # BatchNorm
                lambda_mult = lambdas[i]
                lambda_div = lambdas[i - 1]

                if lambda_mult is not None:
                    mod_weight = mod_weight * lambda_mult

                mass += torch.sum(mod_weight)

            elif mod_weight.ndim == 2:  # Linear
                lambda_mult = lambdas[i]
                lambda_div = lambdas[i - 1]
                if lambda_div is not None:
                    mod_weight = mod_weight @ torch.diag(1 / lambda_div)

                if lambda_mult is not None:
                    mod_weight = torch.diag(lambda_mult) @ mod_weight

                mass += torch.sum(mod_weight)
            else:
                lambda_mult = lambdas[i]
                lambda_div = lambdas[i - 1]
                if lambda_div is not None:
                    mod_weight = mod_weight.transpose(0, 1).view(mod_weight.shape[1], -1) / lambda_div[None, :, None]

                if lambda_mult is not None:
                    mod_weight = mod_weight * lambda_mult[:, None, None]

                mass += torch.sum(mod_weight)
        return mass

    def _get_weights(self, model, squared=True):
        weights = []
        for _key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                weight = (mod.weight.view(mod.out_channels, mod.in_channels // mod.groups, -1).detach() ** (2 if squared else 1)).sum(-1)
                if mod.groups == 2:
                    dim = mod.out_channels // mod.groups
                    weight = weight.reshape(mod.groups, dim).repeat(1, mod.groups)
                    mask = torch.stack(
                        [
                            torch.cat([torch.ones(dim), torch.zeros(dim)]),
                            torch.cat([torch.zeros(dim), torch.ones(dim)]),
                        ]
                    )
                    weight = (weight * mask).transpose(1, 0)
                elif mod.groups != 1:
                    raise NotImplementedError("Only groups=1 or 2 supported")
                weights.append(weight)
            elif isinstance(mod, torch.nn.modules.Linear | torch.nn.modules.batchnorm.BatchNorm2d):
                weights.append(mod.weight.detach() ** (2 if squared else 1))
            else:
                pass
        return weights

    def _scale_conv(
        self,
        module: torch.nn.modules.Conv2d,
        lambda_mult: torch.Tensor = None,
        lambda_div: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """Scale the weights of a convolutional layer.

        Args:
            module (torch.nn.modules.Conv2d): The convolutional layer to scale.
            lambda_mult (torch.Tensor, optional): The scaling factors for the weights. Defaults to
                None.
            lambda_div (torch.Tensor, optional): The scaling factors for the channels. Defaults to
                None.

        Returns:
            List[torch.Tensor]: The scaled weights.
        """
        out_ch = module.out_channels
        groups = module.groups
        weight = torch.clone(module.weight.detach())

        if lambda_mult is not None:
            weight *= lambda_mult[:, None, None, None]

        if lambda_div is not None:
            if groups != 1:
                for g in range(groups):
                    weight[out_ch // groups * g : out_ch // groups * (g + 1), :] /= lambda_div[g]
            else:
                weight /= lambda_div[None, :, None, None]

        if module.bias is not None:
            bias = torch.clone(module.bias.detach())
            bias *= lambda_mult if lambda_mult is not None else 1
            weight = [weight, bias]
        else:
            weight = [weight]

        return weight

    def _scale_linear(
        self,
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

    def _scale_bn(
        self,
        module: torch.nn.modules.batchnorm.BatchNorm2d,
        lambda_mult: torch.Tensor = None,
        lambda_div: torch.Tensor = None,
    ) -> list[torch.Tensor]:
        """Scale the weights of a bn layer.

        Args:
            module (torch.nn.modules.batchnorm.BatchNorm2d): The bn layer to scale.
            lambda_mult (torch.Tensor, optional): The scaling factors for the weights. Defaults to
                None.
            lambda_div (torch.Tensor, optional): The scaling factors for the channels. Defaults to
                None.

        Returns:
            List[torch.Tensor]: The scaled weights.
        """
        running_mean = module.running_mean.detach().clone()
        running_var = module.running_var.detach().clone()
        num_batches_tracked = module.num_batches_tracked.detach().clone()

        weight = module.weight.detach().clone()

        if lambda_div is not None:
            running_mean *= lambda_div
            running_var *= lambda_div**2

        if lambda_mult is not None:
            weight *= lambda_mult

        if module.bias is not None:
            bias = module.bias.detach().clone()
            bias *= lambda_mult if lambda_mult is not None else 1
            weight = [running_mean, running_var, num_batches_tracked, weight, bias]
        else:
            weight = [running_mean, running_var, num_batches_tracked, weight]

        return weight

    def change_bn_eps(self, model, eps_value: float = 0.0):
        for _key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                mod.eps = eps_value
            else:
                pass
        return model
