# A symmetry-aware exploration of Bayesian posteriors

This is the official repository for the code of the paper [A Symmetry-Aware Exploration of Bayesian Neural Network Posteriors](https://arxiv.org/abs/2310.08287) published at ICLR 2024.

## The Checkpoint dataset

The Checkpoint dataset can be found on [Hugging Face](https://huggingface.co/datasets/torch-uncertainty/Checkpoints). I have put there the most interesting networks and plan to add the rest shortly. Tell me if this is of interest to you in an issue.

The a large part of the models were trained using the [TorchUncertainty](https://github.com/ENSTA-U2IS-AI/torch-uncertainty).

## The code

The code currently includes the scripts to "remove" permutation and scaling symmetries, including the convex optimization using the `cvxpy` package. The whole code of the experiments is still a bit messy but I plan to add it here.

The translation of [mmdagg](https://github.com/antoninschrab/mmdagg-paper) to PyTorch can be found [here](https://github.com/o-laurent/torchMMD).

## The poster

I also provide the (new) vertical poster of the paper.

## Something missing?

:construction: I'm currently working on cleaning up the rest of the code, so raise an issue if you would like it to be prioritized.