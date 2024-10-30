# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pylint: skip-file
# pytype: skip-file
"""Various sampling methods."""
from ast import Not
import functools
import torch
import numpy as np
import abc
from models.model_utils import from_flattened_numpy, to_flattened_numpy, get_score_fn
from scipy import integrate
import sde_lib
from models import model_utils as mutils
from utils.utils import *
from absl import flags

FLAGS = flags.FLAGS
_CORRECTORS = {}
_PREDICTORS = {}


def register_predictor(cls=None, *, name=None):
    """A decorator for registering predictor classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _PREDICTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _PREDICTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def register_corrector(cls=None, *, name=None):
    """A decorator for registering corrector classes."""

    def _register(cls):
        if name is None:
            local_name = cls.__name__
        else:
            local_name = name
        if local_name in _CORRECTORS:
            raise ValueError(
                f'Already registered model with name: {local_name}')
        _CORRECTORS[local_name] = cls
        return cls

    if cls is None:
        return _register
    else:
        return _register(cls)


def get_predictor(name):
    return _PREDICTORS[name]


def get_corrector(name):
    return _CORRECTORS[name]


def get_sampling_fn(config, sde, shape, inverse_scaler, eps):
    """Create a sampling function.

    Args:
      config: A `ml_collections.ConfigDict` object that contains all configuration information.
      sde: A `sde_lib.SDE` object that represents the forward SDE.
      shape: A sequence of integers representing the expected shape of a single sample.
      inverse_scaler: The inverse data normalizer function.
      eps: A `float` number. The reverse-time SDE is only integrated to `eps` for numerical stability.

    Returns:
      A function that takes random states and a replicated training state and outputs samples with the
        trailing dimensions matching `shape`.
    """

    sampler_name = config.sampling.method
    # Probability flow ODE sampling with black-box ODE solvers
    if sampler_name.lower() == 'ode':
        raise NotImplementedError
    # Predictor-Corrector sampling. Predictor-only and Corrector-only samplers are special cases.
    elif sampler_name.lower() == 'pc':
        predictor = get_predictor(config.sampling.predictor.lower())
        corrector = get_corrector(config.sampling.corrector.lower())
        sampling_fn = get_pc_sampler(config=config,
                                     sde=sde,
                                     shape=shape,
                                     predictor=predictor,
                                     corrector=corrector,
                                     inverse_scaler=inverse_scaler,
                                     snr=config.sampling.snr,
                                     corrector_mse=config.sampling.corrector_mse,
                                     channel_merge=config.model.channel_merge, 
                                     n_steps=config.sampling.n_steps_each,
                                     probability_flow=config.sampling.probability_flow,
                                     continuous=config.training.continuous,
                                     denoise=config.sampling.noise_removal,
                                     eps=eps,
                                     device=config.device)
    else:
        raise ValueError(f"Sampler name {sampler_name} unknown.")

    return sampling_fn


class Predictor(abc.ABC):
    """The abstract class for a predictor algorithm."""

    def __init__(self, sde, pet_score_fn, mri_score_fn, probability_flow=False):
        super().__init__()
        self.sde = sde
        # Compute the reverse SDE/ODE
        self.rsde = sde.reverse(pet_score_fn, mri_score_fn, probability_flow)
        self.pet_score_fn = pet_score_fn
        self.mri_score_fn = mri_score_fn

    @abc.abstractmethod
    def update_fn(self, x, t, mri):
        """One update of the predictor.

        Args:
          x: A PyTorch tensor representing the current state
          t: A Pytorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


class Corrector(abc.ABC):
    """The abstract class for a corrector algorithm."""

    def __init__(self, sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps):
        super().__init__()
        self.sde = sde
        self.pet_score_fn = pet_score_fn
        self.mri_score_fn = mri_score_fn
        self.snr = snr
        self.corrector_mse = corrector_mse
        self.channel_merge = channel_merge
        self.n_steps = n_steps

    @abc.abstractmethod
    def update_fn(self, x, t, mri):
        """One update of the corrector.

        Args:
          x: A PyTorch tensor representing the current state
          t: A PyTorch tensor representing the current time step.

        Returns:
          x: A PyTorch tensor of the next state.
          x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
        """
        pass


@register_predictor(name='euler_maruyama')
class EulerMaruyamaPredictor(Predictor):
    def __init__(self, sde, pet_score_fn, mri_score_fn, probability_flow=False):
        super().__init__(sde, pet_score_fn, mri_score_fn, probability_flow)

    def update_fn(self, x, t, mri):
        if isinstance(self.sde, sde_lib.PETSDE):
            x, x_mean = self.rsde.sde(x, t, mri)
        else:
            dt = -1. / self.rsde.N # 就是在离散化，就是delta t, reverse diffusion的dt在beta里
            z = torch.randn_like(x)
            drift, diffusion = self.rsde.sde(x, t, mri)
            x_mean = x + drift * dt
            x = x_mean + diffusion[:, None, None, None] * np.sqrt(-dt) * z
        return x, x_mean


@register_predictor(name='reverse_diffusion')
class ReverseDiffusionPredictor(Predictor):
    def __init__(self, sde, pet_score_fn, mri_score_fn, probability_flow=False):
        super().__init__(sde, pet_score_fn, mri_score_fn, probability_flow)

    def update_fn(self, x, t, mri):
        if isinstance(self.sde, sde_lib.PETSDE):
            z = torch.randn_like(x)
            x, x_mean = self.rsde.discretize(x, t, mri)
        else:
            f, G = self.rsde.discretize(x, t, mri)
            z = torch.randn_like(x)
            x_mean = x - f
            x = x_mean + G[:, None, None, None] * z
        return x, x_mean


@register_predictor(name='none')
class NonePredictor(Predictor):
    """An empty predictor that does nothing."""

    def __init__(self, sde, pet_score_fn, mri_score_fn, probability_flow=False):
        pass

    def update_fn(self, x, t, mri):
        return x, x


@register_corrector(name='langevin')
class LangevinCorrector(Corrector):
    def __init__(self, sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps):
        super().__init__(sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps)
        if not isinstance(sde, sde_lib.VPSDE) \
                and not isinstance(sde, sde_lib.VESDE) \
                and not isinstance(sde, sde_lib.subVPSDE) \
                and not isinstance(sde, sde_lib.PETSDE):
            raise NotImplementedError(
                f"SDE class {sde.__class__.__name__} not yet supported.")

    def update_fn(self, x, t, mri):
        sde = self.sde
        pet_score_fn = self.pet_score_fn
        mri_score_fn = self.mri_score_fn
        n_steps = self.n_steps
        target_snr = self.snr
        corrector_mse = self.corrector_mse
        channel_merge = self.channel_merge

        if isinstance(sde, sde_lib.VPSDE) or isinstance(sde, sde_lib.subVPSDE):
            timestep = (t * (sde.N - 1) / sde.T).long()
            alpha = sde.alphas.to(t.device)[timestep]
        else:
            alpha = torch.ones_like(t)

        for i in range(n_steps):#1
            x = torch.cat((x, mri), 1)
            grad = pet_score_fn(x, t)
            if not channel_merge:
                grad = grad[:, 0, :, :]
                grad = torch.unsqueeze(grad, 1)
            
            x = x[:, 0, :, :]
            x = torch.unsqueeze(x, 1)
            noise = torch.randn_like(x)
            grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
            step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha

            x_mean = x + step_size[:, None, None, None] * grad
            x = x_mean + torch.sqrt(step_size * 2)[:, None, None, None] * noise

        return x, x_mean


@register_corrector(name='none')
class NoneCorrector(Corrector):
    """An empty corrector that does nothing."""

    def __init__(self, sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps):
        pass

    def update_fn(self, x, t, mri):
        return x, x


def shared_predictor_update_fn(x, t, mri, sde, pet_model, mri_model, predictor, probability_flow, continuous):
    """A wrapper that configures and returns the update function of predictors."""
    pet_score_fn = mutils.get_score_fn(sde, pet_model, train=False, continuous=continuous)
    mri_score_fn = mutils.get_score_fn(sde, mri_model, train=False, continuous=continuous)
    if predictor is None:
        # Corrector-only sampler
        predictor_obj = NonePredictor(sde, pet_score_fn, mri_score_fn, probability_flow)
    else:
        predictor_obj = predictor(sde, pet_score_fn, mri_score_fn, probability_flow)
    return predictor_obj.update_fn(x, t, mri)


def shared_corrector_update_fn(x, t, mri, sde, pet_model, mri_model, corrector, continuous, snr, corrector_mse, channel_merge, n_steps):
    """A wrapper tha configures and returns the update function of correctors."""
    pet_score_fn = mutils.get_score_fn(sde, pet_model, train=False, continuous=continuous)
    mri_score_fn = mutils.get_score_fn(sde, mri_model, train=False, continuous=continuous)
    if corrector is None:
        # Predictor-only sampler
        corrector_obj = NoneCorrector(sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps)
    else:
        corrector_obj = corrector(sde, pet_score_fn, mri_score_fn, snr, corrector_mse, channel_merge, n_steps)
    return corrector_obj.update_fn(x, t, mri)


def get_pc_sampler(config, sde, shape, predictor, corrector, inverse_scaler, snr, corrector_mse,
                   channel_merge, n_steps=1, probability_flow=False, continuous=False,
                   denoise=True, eps=1e-3, device='cuda'):
    """Create a Predictor-Corrector (PC) sampler.

    Args:
      sde: An `sde_lib.SDE` object representing the forward SDE.
      shape: A sequence of integers. The expected shape of a single sample.
      predictor: A subclass of `sampling.Predictor` representing the predictor algorithm.
      corrector: A subclass of `sampling.Corrector` representing the corrector algorithm.
      inverse_scaler: The inverse data normalizer.
      snr: A `float` number. The signal-to-noise ratio for configuring correctors.
      n_steps: An integer. The number of corrector steps per predictor update.
      probability_flow: If `True`, solve the reverse-time probability flow ODE when running the predictor.
      continuous: `True` indicates that the score model was continuously trained.
      denoise: If `True`, add one-step denoising to the final samples.
      eps: A `float` number. The reverse-time SDE and ODE are integrated to `epsilon` to avoid numerical issues.
      device: PyTorch device.

    Returns:
      A sampling function that returns samples and the number of function evaluations during sampling.
    """
    # Create predictor & corrector update functions
    predictor_update_fn = functools.partial(shared_predictor_update_fn,
                                            sde=sde,
                                            predictor=predictor,
                                            probability_flow=probability_flow,
                                            continuous=continuous)
    corrector_update_fn = functools.partial(shared_corrector_update_fn,
                                            sde=sde,
                                            corrector=corrector,
                                            channel_merge=channel_merge, 
                                            continuous=continuous,
                                            snr=snr,
                                            corrector_mse=corrector_mse,
                                            n_steps=n_steps)

    def pc_sampler(pet_model, mri_model, mri):
        """ The PC sampler funciton.

        Args:
          model: A score model.
        Returns:
          Samples, number of function evaluations.
        """
        with torch.no_grad():
            # Initial sample
            x = sde.prior_sampling(shape).to(device) # noise
            
            timesteps = torch.linspace(sde.T, eps, sde.N, device=device)

            for i in range(sde.N):#1000
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                x, x_mean = corrector_update_fn(x, vec_t, mri, pet_model=pet_model, mri_model=mri_model)
                x = x.type(torch.FloatTensor).to(device)
                x_mean = x_mean.type(torch.FloatTensor).to(device)
                x, x_mean = predictor_update_fn(x, vec_t, mri, pet_model=pet_model, mri_model=mri_model)
                x = x.type(torch.FloatTensor).to(device)
                x_mean = x_mean.type(torch.FloatTensor).to(device)
                # if (i + 1) % 100 == 0:
                #     save_mat(FLAGS.config.sampling.folder, x_mean, 'x_mean', i, normalize=False)

            return inverse_scaler(x_mean if denoise else x)

    return pc_sampler