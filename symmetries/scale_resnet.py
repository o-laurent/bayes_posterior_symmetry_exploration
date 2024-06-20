import copy

import cvxpy as cp
import numpy as np
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


class Scaler:
    integrity_tol = 1e-4

    def __init__(
        self,
        input_shape: tuple,
        verbose: bool = False,
    ):
        self.inputs = torch.randn(3, *input_shape).double()
        self.verbose = verbose

    def _scale_conv(
        self,
        module: torch.nn.modules.Conv2d,
        lambda_mult: torch.Tensor = None,
        lambda_div: torch.Tensor = None,
        modify_in_place: bool = False,
    ):
        weight = module.weight.detach()
        if not modify_in_place:
            weight = torch.clone(weight)

        if lambda_mult is not None:
            weight *= lambda_mult[:, None, None, None]

        if lambda_div is not None:
            weight /= lambda_div[None, :, None, None]

        if module.bias is not None:
            bias = module.bias.detach()
            if not modify_in_place:
                bias = torch.clone(bias)
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
        modify_in_place: bool = False,
    ):
        weight = module.weight.detach()
        if not modify_in_place:
            weight = torch.clone(weight)

        if lambda_div is not None:
            weight = col_wise_multiplication(weight, 1 / lambda_div)

        if lambda_mult is not None:
            weight = row_wise_multiplication(weight, lambda_mult)

        if module.bias is not None:
            bias = module.bias.detach()
            if not modify_in_place:
                bias = torch.clone(bias)
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
        modify_in_place: bool = False,
    ):
        running_mean = module.running_mean.detach().clone()
        running_var = module.running_var.detach().clone()
        num_batches_tracked = module.num_batches_tracked.detach().clone()

        weight = module.weight.detach()
        if not modify_in_place:
            weight = torch.clone(weight)

        if lambda_div is not None:
            running_mean *= lambda_div
            running_var *= lambda_div**2

        if lambda_mult is not None:
            weight *= lambda_mult

        if module.bias is not None:
            bias = module.bias.detach()
            if not modify_in_place:
                bias = torch.clone(bias)
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

    def _check_consistency(self, model, scaled_weights):
        model.eval()
        model = self.change_bn_eps(model)
        model_hash = model(self.inputs).detach()
        scaled_model = copy.deepcopy(model)
        scaled_model.eval()
        scaled_model.load_state_dict(scaled_weights)
        scaled_model = self.change_bn_eps(scaled_model)
        scaled_hash = scaled_model(self.inputs)
        mse = torch.sum((scaled_hash - model_hash) ** 2).item()
        success = mse < self.integrity_tol
        scaled_weights = None
        sub_weights = self._get_weights(scaled_model)
        ending_mass = sum([torch.sum(w) for w in sub_weights]) if success else None
        if self.verbose:
            print(f"Model MSE:{mse}")
            if success:
                print(f"Ending L2 mass {ending_mass}")
            else:
                print("Model integrity not preserved")
        return ending_mass

    def _get_weights(self, model, squared=True):
        weights = []
        for _key, mod in model.named_modules():
            if isinstance(mod, torch.nn.modules.Conv2d):
                weight = (mod.weight.view(mod.out_channels, mod.in_channels, -1).detach() ** (2 if squared else 1)).sum(-1)
                weights.append(weight)
            elif isinstance(mod, torch.nn.modules.Linear | torch.nn.modules.batchnorm.BatchNorm2d):
                weight = mod.weight.detach() ** (2 if squared else 1)
                weights.append(weight)
            else:
                pass
        return weights

    @torch.no_grad()
    def _scale_stats(
        self,
        model,
        method: str = "unitnorm",
        same_mult_constraints: dict | None = None,
        same_div_constraints: dict | None = None,
    ):
        if same_div_constraints is None:
            same_div_constraints = {}
        if same_mult_constraints is None:
            same_mult_constraints = {}
        model = copy.deepcopy(model).double()
        if self.verbose:
            sub_mass = self._get_weights(model)
            print(f"Initial mass: { sum([torch.sum(w) for w in sub_mass])}")

        lambda_mult_dict = {}
        lambda_div_dict = {}
        previous_key = None
        scaled_weights = {}
        for key, mod in model.named_modules():
            if isinstance(
                mod,
                torch.nn.modules.Conv2d | torch.nn.modules.Linear | torch.nn.modules.batchnorm.BatchNorm2d,
            ):
                lambda_div = None
                if previous_key is not None:
                    lambda_div = lambda_mult_dict[previous_key].clone()

                if key in same_div_constraints:
                    equivalent_module = same_div_constraints[key]
                    lambda_div = lambda_div_dict[equivalent_module].clone()

                lambda_div_dict[key] = lambda_div
            else:
                continue

            # Define lambda_mult and lambda_div
            if isinstance(mod, torch.nn.modules.Conv2d):
                if lambda_div is not None:
                    if method == "unitvar":
                        lambda_mult = 1 / self._scale_conv(
                            mod,
                            lambda_mult=None,
                            lambda_div=lambda_div,
                            modify_in_place=False,
                        )[0].view(mod.out_channels, -1).std(dim=-1)
                    elif method == "unitnorm":
                        lambda_mult = 1 / torch.norm(
                            self._scale_conv(
                                mod,
                                lambda_mult=None,
                                lambda_div=lambda_div,
                                modify_in_place=False,
                            )[0].view(mod.out_channels, -1),
                            p=2,
                            dim=-1,
                        )
                else:
                    if method == "unitvar":
                        lambda_mult = 1 / mod.weight.view(mod.out_channels, -1).std(dim=-1)
                    elif method == "unitnorm":
                        lambda_mult = 1 / torch.norm(mod.weight.view(mod.out_channels, -1), p=2, dim=-1)

            elif isinstance(mod, torch.nn.modules.Linear):
                if method == "unitvar":
                    lambda_mult = 1 / col_wise_multiplication(mod.weight.detach().clone(), 1 / lambda_div).std(dim=1)
                elif method == "unitnorm":
                    lambda_mult = 1 / torch.norm(
                        col_wise_multiplication(mod.weight.detach().clone(), 1 / lambda_div),
                        p=2,
                        dim=1,
                    )

            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                lambda_mult = (
                    1
                    / self._scale_bn(
                        copy.deepcopy(mod),
                        lambda_mult=None,
                        lambda_div=lambda_div,
                        modify_in_place=False,
                    )[3]
                )
            else:
                continue

            # Overwrite with constraints
            if key in same_mult_constraints:
                equivalent_module = same_mult_constraints[key]
                lambda_mult = None if equivalent_module is None else lambda_mult_dict[equivalent_module].clone()

            lambda_mult_dict[key] = lambda_mult

            # Apply changes
            if isinstance(mod, torch.nn.modules.Conv2d):
                cweight = self._scale_conv(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                lambda_mult_dict[key] = lambda_mult

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]
                previous_key = key

            elif isinstance(mod, torch.nn.modules.Linear):
                cweight = self._scale_linear(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                lambda_mult_dict[key] = lambda_mult

                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]
                previous_key = key

            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                (
                    running_mean,
                    running_var,
                    num_batches_tracked,
                    weight,
                    bias,
                ) = self._scale_bn(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                lambda_mult_dict[key] = lambda_mult
                scaled_weights[key + ".running_mean"] = running_mean.clone()
                scaled_weights[key + ".running_var"] = running_var.clone()
                scaled_weights[key + ".num_batches_tracked"] = num_batches_tracked.clone()

                scaled_weights[key + ".weight"] = weight.clone()
                if bias is not None:
                    scaled_weights[key + ".bias"] = bias.clone()
                previous_key = key

        self._check_consistency(model, scaled_weights)
        return scaled_weights, lambda_mult_dict, lambda_div_dict

    @torch.no_grad()
    def _apply_lmbdas(
        self,
        model,
        lambda_mult_dict,
        lambda_div_dict,
    ):
        if self.verbose:
            sub_mass = self._get_weights(model)
            print(f"Initial mass: { sum([torch.sum(w) for w in sub_mass])}")

        scaled_weights = {}
        for key, mod in model.named_modules():
            if isinstance(
                mod,
                torch.nn.modules.Conv2d | torch.nn.modules.Linear | torch.nn.modules.batchnorm.BatchNorm2d,
            ):
                lambda_div = lambda_div_dict[key]
                lambda_mult = lambda_mult_dict[key]
            else:
                continue

            # Apply changes
            if isinstance(mod, torch.nn.modules.Conv2d):
                cweight = self._scale_conv(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

            elif isinstance(mod, torch.nn.modules.Linear):
                cweight = self._scale_linear(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                scaled_weights[key + ".weight"] = cweight[0]
                if len(cweight) > 1:
                    scaled_weights[key + ".bias"] = cweight[1]

            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                (
                    running_mean,
                    running_var,
                    num_batches_tracked,
                    weight,
                    bias,
                ) = self._scale_bn(
                    mod,
                    lambda_mult=lambda_mult,
                    lambda_div=lambda_div,
                    modify_in_place=False,
                )
                scaled_weights[key + ".running_mean"] = running_mean
                scaled_weights[key + ".running_var"] = running_var
                scaled_weights[key + ".num_batches_tracked"] = num_batches_tracked

                scaled_weights[key + ".weight"] = weight.clone()
                if bias is not None:
                    scaled_weights[key + ".bias"] = bias.clone()

        self._check_consistency(model, scaled_weights)
        return scaled_weights

    @torch.no_grad()  # prendre en compte les residual
    def _scale_cvx_optimize(self, model, same_mult_constraints, same_div_constraints, verbose=True):
        mass, lambda_mult_dict, lambda_div_dict, terms = self._create_cvxpy_problem(model, same_mult_constraints, same_div_constraints)
        prob = cp.Problem(cp.Minimize(mass))
        prob.solve(verbose=verbose)
        if prob.status != "optimal":
            raise RuntimeError("Optimization failed")
        final_lambda_mult_dict = {}
        for k, v in lambda_mult_dict.items():
            if v is not None:
                final_lambda_mult_dict[k] = torch.exp(torch.tensor(v.value))
            else:
                final_lambda_mult_dict[k] = None
        final_lambda_div_dict = {}
        for k, v in lambda_div_dict.items():
            if v is not None:
                final_lambda_div_dict[k] = torch.exp(torch.tensor(v.value))
            else:
                final_lambda_div_dict[k] = None
        scaled_weights = self._apply_lmbdas(model, final_lambda_mult_dict, final_lambda_div_dict)
        return final_lambda_mult_dict, final_lambda_div_dict, scaled_weights

    @torch.no_grad()  # prendre en compte les residual
    def _create_cvxpy_problem(self, model, same_mult_constraints: dict | None = None, same_div_constraints: dict | None = None):
        if same_div_constraints is None:
            same_div_constraints = {}
        if same_mult_constraints is None:
            same_mult_constraints = {}
        lambda_mult_dict = {}
        lambda_div_dict = {}
        previous_key = None
        terms = []
        for key, mod in model.named_modules():
            lambda_div = None
            if previous_key is not None:
                lambda_div = lambda_mult_dict[previous_key]
            if key in same_div_constraints:
                equivalent_module = same_div_constraints[key]
                lambda_div = lambda_div_dict[equivalent_module]
            lambda_div_dict[key] = lambda_div

            if isinstance(mod, torch.nn.modules.Conv2d):
                lambda_mult = cp.Variable(mod.out_channels)
            elif isinstance(mod, torch.nn.modules.Linear):
                lambda_mult = cp.Variable(mod.out_features)
            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                lambda_mult = cp.Variable(mod.num_features)
            else:
                continue

            if key in same_mult_constraints:
                equivalent_module = same_mult_constraints[key]
                lambda_mult = None if equivalent_module is None else lambda_mult_dict[equivalent_module]
            lambda_mult_dict[key] = lambda_mult

            # Apply changes
            if isinstance(mod, torch.nn.modules.Conv2d):
                mod_weight = (mod.weight.view(mod.out_channels, mod.in_channels, -1).detach().clone() ** 2).sum(-1)
                if lambda_div is not None and lambda_mult is None:
                    mod_weight = mod_weight @ cp.diag(cp.exp(-2 * lambda_div))
                elif lambda_mult is not None and lambda_div is None:
                    mod_weight = cp.diag(cp.exp(2 * lambda_mult)) @ mod_weight
                elif lambda_mult is not None and lambda_div is not None:
                    tmp_div = np.ones((mod.out_channels, 1)) @ cp.reshape(lambda_div, (1, mod.in_channels))
                    tmp_mult = cp.reshape(lambda_mult, (mod.out_channels, 1)) @ np.ones((1, mod.in_channels))
                    omega_scale = cp.exp(2 * (tmp_mult - tmp_div))
                    mod_weight = cp.multiply(mod_weight, omega_scale)
                previous_key = key
                terms.append(cp.sum(mod_weight))

            elif isinstance(mod, torch.nn.modules.Linear):
                mod_weight = mod.weight.detach().clone() ** 2
                mod_weight = mod_weight @ cp.diag(cp.exp(-2 * lambda_div))
                previous_key = key
                terms.append(cp.sum(mod_weight))

            elif isinstance(mod, torch.nn.modules.batchnorm.BatchNorm2d):
                mod_weight = mod.weight.detach().clone() ** 2
                mod_weight = cp.diag(cp.exp(2 * lambda_mult)) @ mod_weight
                terms.append(cp.sum(mod_weight))
                previous_key = key

        return cp.sum(terms), lambda_mult_dict, lambda_div_dict, terms
