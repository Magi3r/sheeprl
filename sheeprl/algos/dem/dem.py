"""Dreamer-V3 implementation from [https://arxiv.org/abs/2301.04104](https://arxiv.org/abs/2301.04104)
Adapted from the original implementation from https://github.com/danijar/dreamerv3
"""

from __future__ import annotations

import copy
import os
import warnings
from functools import partial
# from typing import Any, Dict, Sequence
from typing import Any, Callable, Dict, Sequence, Tuple
import time

import gymnasium as gym
import hydra
import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric import Fabric
from lightning.fabric.wrappers import _FabricModule
from torch import Tensor
from torch.distributions import Distribution, Independent, OneHotCategorical
from torch.optim import Optimizer
from torchmetrics import SumMetric
from tqdm import tqdm

from sheeprl.algos.dreamer_v2.utils import compute_stochastic_state

from sheeprl.algos.dem.agent import WorldModel, build_agent
from sheeprl.algos.dem.loss import reconstruction_loss, reconstruction_loss_rehearsal
from sheeprl.algos.dem.utils import Moments, compute_lambda_values, prepare_obs, test
from sheeprl.data.buffers import EnvIndependentReplayBuffer, SequentialReplayBuffer
from sheeprl.envs.wrappers import RestartOnException
from sheeprl.utils.distribution import (
    BernoulliSafeMode,
    MSEDistribution,
    SymlogDistribution,
    TwoHotEncodingDistribution,
)
from sheeprl.utils.env import make_env
from sheeprl.utils.logger import get_log_dir, get_logger
from sheeprl.utils.metric import MetricAggregator
from sheeprl.utils.registry import register_algorithm
from sheeprl.utils.timer import timer
from sheeprl.utils.utils import Ratio, save_configs

import pkgutil

# import os
# os.environ["HYDRA_FULL_ERROR"] = "1"

from sheeprl.algos.dem.utils import parallel_additive_correction_delta
# from sheeprl.algos.dem.episodic_memory import EpisodicMemory as EM
from sheeprl.algos.dem.episodic_memory_gpu import GPUEpisodicMemory as EM

# Decomment the following two lines if you cannot start an experiment with DMC environments
# os.environ["PYOPENGL_PLATFORM"] = ""
# os.environ["MUJOCO_GL"] = "osmesa"

def dynamic_learning(
    world_model: WorldModel,
    data: Dict[str, Tensor],
    batch_actions: Tensor,
    embedded_obs: Dict[str, Tensor],
    stochastic_size: int,
    discrete_size: int,
    recurrent_state_size: int,
    batch_size: int,
    sequence_length: int,
    decoupled_rssm: bool,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size
    recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)

    if decoupled_rssm:
        posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
        for i in range(0, sequence_length):
            if i == 0:
                posterior = torch.zeros_like(posteriors[:1])
            else:
                posterior = posteriors[i - 1 : i]
            recurrent_state, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
    else:
        posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
        posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
        posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
        for i in range(0, sequence_length):
            recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
                posterior,
                recurrent_state,
                batch_actions[i : i + 1],
                embedded_obs[i : i + 1],
                data["is_first"][i : i + 1],
            )
            recurrent_states[i] = recurrent_state
            priors_logits[i] = prior_logits
            posteriors[i] = posterior
            posteriors_logits[i] = posterior_logits
    latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
    return latent_states, priors_logits, posteriors_logits, posteriors, recurrent_states


def behaviour_learning(
    posteriors: torch.Tensor,
    recurrent_states: torch.Tensor,
    posteriors_logits: torch.Tensor,
    data: Dict[str, torch.Tensor],
    world_model: WorldModel,
    actor: _FabricModule,
    stoch_state_size: int,
    recurrent_state_size: int,
    batch_size: int,
    sequence_length: int,
    horizon: int,
    device: torch.device,
    episodic_memory: EM | None,
    read_dream_mean_std: torch.tensor,
    read_z: float = 1.0,
    use_acd: bool = False,
    adc_weighting: bool = False,
    k_neighbors: int = 10,
    read_exp_mov_avg_alpha: float = 0.99, ## TOIDO: default value make sense?
) -> Tuple[torch.Tensor, torch.Tensor]:
    imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)               ## ? wird später in imagination überschrieben ~Josch
    recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    imagined_trajectories = torch.empty(
        horizon + 1,
        batch_size * sequence_length,
        stoch_state_size + recurrent_state_size,
        device=device,
    )
    imagined_trajectories[0] = imagined_latent_state
    imagined_actions = torch.empty(
        horizon+ 1,
        batch_size * sequence_length,
        data["actions"].shape[-1],
        device=device,
    )
    actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    imagined_actions[0] = actions

    prev_imagined_prior_logits = posteriors_logits.detach().reshape(1, -1, stoch_state_size) ## torch.empty_like(imagined_prior, device=device)
    prev_recurrent_state = torch.empty_like(recurrent_state, device=device)
    prev_recurrent_state.copy_(recurrent_state)
    for i in range(1, horizon + 1):    ## lopin 15 times
        ## TODO: Assumption: currently these values get detached before loss backward, so no WorldModel is trained here 
        ##   (so we could combine both _transition calls in imagination for faster inference (one call for actual value, one only for uncertainties))
        if episodic_memory is not None:
            imagined_prior_logits, recurrent_state, uncertainties = world_model.rssm.imagination(imagined_prior, recurrent_state, actions, \
                                                                                          return_logits=True, return_uncertainty = True)
        else:
            imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions, return_uncertainty = False)
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)   ## after: -> [1, 1024, 1024]
        ### update treshold based on rolling mean and std ~0.145592ms
        if episodic_memory is not None:
            read_dream_mean_std[0] = torch.mean(uncertainties) * read_exp_mov_avg_alpha + read_dream_mean_std[0] * (1 - read_exp_mov_avg_alpha)           
            read_dream_mean_std[1] = torch.std(uncertainties) * read_exp_mov_avg_alpha + read_dream_mean_std[1] * (1 - read_exp_mov_avg_alpha)
            lookup_treshold = read_dream_mean_std[0] + read_z * read_dream_mean_std[1]

            if use_acd:
                k: int = k_neighbors

                uncertainty_mask = uncertainties > lookup_treshold          ## shape [1024]

                ## ACDs: [num_uncertainties>_threashold, 1, 1024]
                ## TODO: AKTUELL: query = (h_t+1, z_t+1, a_t)
                ACDs: torch.tensor = parallel_additive_correction_delta(prev_recurrent_state[:, uncertainty_mask, :], 
                                                                    prev_imagined_prior_logits[:, uncertainty_mask, :], 
                                                                    actions[:, uncertainty_mask, :], 
                                                                    episodic_memory, 
                                                                    world_model, 
                                                                    k, 
                                                                    device=device,
                                                                    adc_weighting=adc_weighting)

                imagined_prior_logits[:, uncertainty_mask, :] += ACDs

                prev_recurrent_state.copy_(recurrent_state)
                prev_imagined_prior_logits.copy_(imagined_prior_logits)

            imagined_prior = compute_stochastic_state(imagined_prior_logits, discrete=world_model.rssm.discrete, sample=True)
            imagined_prior = imagined_prior.view(1, -1, stoch_state_size)

        imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
        imagined_trajectories[i] = imagined_latent_state
        actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
        imagined_actions[i] = actions
    ##############################################################################################################################################################

    # imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)
    # recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    # imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    # imagined_trajectories = torch.empty(
    #     horizon + 1,
    #     batch_size * sequence_length,
    #     stoch_state_size + recurrent_state_size,
    #     device=device,
    # )
    # imagined_trajectories[0] = imagined_latent_state
    # imagined_actions = torch.empty(
    #     horizon + 1,
    #     batch_size * sequence_length,
    #     data["actions"].shape[-1],
    #     device=device,
    # )
    # actions_list, _ = actor(imagined_latent_state.detach())
    # actions = torch.cat(actions_list, dim=-1)
    # imagined_actions[0] = actions

    # # Imagine trajectories in the latent space
    # for i in range(1, horizon + 1):
    #     imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
    #     imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
    #     imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    #     imagined_trajectories[i] = imagined_latent_state
    #     actions_list, _ = actor(imagined_latent_state.detach())
    #     actions = torch.cat(actions_list, dim=-1)
    #     imagined_actions[i] = actions

    return imagined_trajectories, imagined_actions

def train(
    fabric: Fabric,
    world_model: WorldModel,
    actor: _FabricModule,
    critic: _FabricModule,
    target_critic: torch.nn.Module,
    world_optimizer: Optimizer,
    actor_optimizer: Optimizer,
    critic_optimizer: Optimizer,
    data: Dict[str, Tensor],
    aggregator: MetricAggregator | None,
    cfg: Dict[str, Any],
    is_continuous: bool,
    actions_dim: Sequence[int],
    moments: Moments,
    compiled_dynamic_learning: Callable,
    compiled_behaviour_learning: Callable,
    compiled_compute_lambda_values: Callable,
    episodic_memory: EM | None,
    read_dream_mean_std: torch.tensor,
    read_z: float = 1.0,
) -> None:
    """Runs one-step update of the agent.

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        actor (_FabricModule): the actor model wrapped with Fabric.
        critic (_FabricModule): the critic model wrapped with Fabric.
        target_critic (nn.Module): the target critic model.
        world_optimizer (Optimizer): the world optimizer.
        actor_optimizer (Optimizer): the actor optimizer.
        critic_optimizer (Optimizer): the critic optimizer.
        data (Dict[str, Tensor]): the batch of data to use for training.
        aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        cfg (DictConfig): the configs.
        is_continuous (bool): whether or not the environment is continuous.
        actions_dim (Sequence[int]): the actions dimension.
        moments (Moments): the moments for normalizing the lambda values.
    """
    # The environment interaction goes like this:
    # Actions:           a0       a1       a2      a4
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     v  /     v  /     v /
    # Observations:  o0       o1       o2       o3
    # Rewards:       0        r1       r2       r3
    # Dones:         0        d1       d2       d3
    # Is-first       1        i1       i2       i3

    batch_size = cfg.algo.per_rank_batch_size
    sequence_length = cfg.algo.per_rank_sequence_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device: torch.device = fabric.device
    batch_obs = {k: data[k] / 255.0 - 0.5 for k in cfg.algo.cnn_keys.encoder}
    batch_obs.update({k: data[k] for k in cfg.algo.mlp_keys.encoder})
    data["is_first"][0, :] = torch.ones_like(data["is_first"][0, :])

    # Given how the environment interaction works, we remove the last actions
    # and add the first one as the zero action
    batch_actions = torch.cat((torch.zeros_like(data["actions"][:1]), data["actions"][:-1]), dim=0)

    # Dynamic Learning
    stoch_state_size = stochastic_size * discrete_size  ## e.g. 32x32

    ##############################################################################################################################################################################
    # recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
    # recurrent_states = torch.empty(sequence_length, batch_size, recurrent_state_size, device=device)
    # priors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
    ##############################################################################################################################################################################

    # Embed observations from the environment
    embedded_obs = world_model.encoder(batch_obs) ## embedded_obs -> torch.Size([64, 16, 12288]

    ##############################################################################################################################################################################
    # ### i think we only use coupled???
    # if cfg.algo.world_model.decoupled_rssm:
    #     posteriors_logits, posteriors = world_model.rssm._representation(embedded_obs)
    #     for i in range(0, sequence_length):
    #         if i == 0:
    #             posterior = torch.zeros_like(posteriors[:1])
    #         else:
    #             posterior = posteriors[i - 1 : i]
    #         recurrent_state, posterior_logits, prior_logits = world_model.rssm.dynamic(
    #             posterior,
    #             recurrent_state,
    #             batch_actions[i : i + 1],
    #             data["is_first"][i : i + 1],
    #         )
    #         recurrent_states[i] = recurrent_state
    #         priors_logits[i] = prior_logits
    # else:
    #     posterior = torch.zeros(1, batch_size, stochastic_size, discrete_size, device=device)
    #     posteriors = torch.empty(sequence_length, batch_size, stochastic_size, discrete_size, device=device)
    #     posteriors_logits = torch.empty(sequence_length, batch_size, stoch_state_size, device=device)
    #     for i in range(0, sequence_length):
    #         ### h, z, z^

    #         recurrent_state, posterior, _, posterior_logits, prior_logits = world_model.rssm.dynamic(
    #             posterior,
    #             recurrent_state,
    #             batch_actions[i : i + 1],
    #             embedded_obs[i : i + 1],
    #             data["is_first"][i : i + 1],
    #         )
    #         # print("THEIR DYNAMICS")
    #         # print("prior shape:", prior_logits.shape)
    #         # print("recurrent_state shape:", recurrent_state.shape)
    #         # print("actions shape:", batch_actions[i : i + 1].shape)
    #         # print("post shape:", posterior.shape)
    #         # # THEIR DYNAMICS
    #         # # prior shape: torch.Size([1, 16, 1024])
    #         # # recurrent_state shape: torch.Size([1, 16, 4096])
    #         # # actions shape: torch.Size([1, 16, 9])
    #         # # post shape: torch.Size([1, 16, 32, 32])

    #         recurrent_states[i] = recurrent_state
    #         priors_logits[i] = prior_logits
    #         posteriors[i] = posterior
    #         posteriors_logits[i] = posterior_logits

    # ### flatten posteriors but only the 32x32, keep (sequence_length, batch_size, 32x32)
    # ### concat with recurrent_states (sequence_length, batch_size, x)
    # latent_states = torch.cat((posteriors.view(*posteriors.shape[:-2], -1), recurrent_states), -1)
    ##############################################################################################################################################################################
    
    # Dynamic Learning
    latent_states, priors_logits, posteriors_logits, posteriors, recurrent_states = compiled_dynamic_learning(
        world_model,
        data,
        batch_actions,
        embedded_obs,
        stochastic_size,
        discrete_size,
        recurrent_state_size,
        batch_size,
        sequence_length,
        cfg.algo.world_model.decoupled_rssm,
        device,
    )

    # Compute predictions for the observations
    reconstructed_obs: Dict[str, torch.Tensor] = world_model.observation_model(latent_states) ## Decoder

    # Compute the distribution over the reconstructed observations
    po = {
        k: MSEDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
        for k in cfg.algo.cnn_keys.decoder
    }
    po.update(
        {
            k: SymlogDistribution(reconstructed_obs[k], dims=len(reconstructed_obs[k].shape[2:]))
            for k in cfg.algo.mlp_keys.decoder
        }
    )

    # Compute the distribution over the rewards
    pr = TwoHotEncodingDistribution(world_model.reward_model(latent_states), dims=1)

    # Compute the distribution over the terminal steps, if required
    pc = Independent(BernoulliSafeMode(logits=world_model.continue_model(latent_states)), 1)
    continues_targets = 1 - data["terminated"]

    # Reshape posterior and prior logits to shape [B, T, 32, 32]
    priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
    posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

    # World model optimization step. Eq. 4 in the paper
    world_optimizer.zero_grad(set_to_none=True)
    rec_loss, kl, state_loss, reward_loss, observation_loss, continue_loss = reconstruction_loss(
        po,
        batch_obs,
        pr,
        data["rewards"],
        priors_logits,
        posteriors_logits,
        cfg.algo.world_model.kl_dynamic,
        cfg.algo.world_model.kl_representation,
        cfg.algo.world_model.kl_free_nats,
        cfg.algo.world_model.kl_regularizer,
        pc,
        continues_targets,
        cfg.algo.world_model.continue_scale_factor,
    )
    fabric.backward(rec_loss)
    world_model_grads = None
    if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
        world_model_grads = fabric.clip_gradients(
            module=world_model,
            optimizer=world_optimizer,
            max_norm=cfg.algo.world_model.clip_gradients,
            error_if_nonfinite=False,
        )
    world_optimizer.step()

    

    # print("Behaviour Learning ~Actor-Critic stuff")
    # Behaviour Learning    ## (Actor-Critic) ~ this shoulnd be important for rehearsal training


    # Behaviour Learning
    imagined_trajectories, imagined_actions = compiled_behaviour_learning(
        posteriors,
        recurrent_states,
        posteriors_logits,
        data,
        world_model,
        actor,
        stoch_state_size,
        recurrent_state_size,
        batch_size,
        sequence_length,
        cfg.algo.horizon,
        device,
        episodic_memory,
        read_dream_mean_std,
        read_z,
        cfg.episodic_memory.use_acd,
        cfg.episodic_memory.adc_weighting,
        cfg.episodic_memory.k_neighbors,
        cfg.episodic_memory.read_exp_mov_avg_alpha
    )

    # ##############################################################################################################################################################################
    # imagined_prior = posteriors.detach().reshape(1, -1, stoch_state_size)               ## ? wird später in imagination überschrieben ~Josch
    # recurrent_state = recurrent_states.detach().reshape(1, -1, recurrent_state_size)
    # imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    # imagined_trajectories = torch.empty(
    #     cfg.algo.horizon + 1,
    #     batch_size * sequence_length,
    #     stoch_state_size + recurrent_state_size,
    #     device=device,
    # )
    # imagined_trajectories[0] = imagined_latent_state
    # imagined_actions = torch.empty(
    #     cfg.algo.horizon + 1,
    #     batch_size * sequence_length,
    #     data["actions"].shape[-1],
    #     device=device,
    # )
    # actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    # imagined_actions[0] = actions

    # # print("their actions:", actions, actions.shape)

    # # The imagination goes like this, with H=3:
    # # Actions:           a'0      a'1      a'2     a'4
    # #                    ^ \      ^ \      ^ \     ^
    # #                   /   \    /   \    /   \   /
    # #                  /     \  /     \  /     \ /
    # # States:        z0 ---> z'1 ---> z'2 ---> z'3
    # # Rewards:       r'0     r'1      r'2      r'3
    # # Values:        v'0     v'1      v'2      v'3
    # # Lambda-values:         l'1      l'2      l'3
    # # Continues:     c0      c'1      c'2      c'3
    # # where z0 comes from the posterior, while z'i is the imagined states (prior)

    # # Imagine trajectories in the latent space
    # # start_time_loop = time.perf_counter_ns()
    # # for i in range(1, cfg.algo.horizon + 1):
    # #     imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions)
    # #     imagined_prior = imagined_prior.view(1, -1, stoch_state_size)
    # #     imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    # #     imagined_trajectories[i] = imagined_latent_state
    # #     actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    # #     imagined_actions[i] = actions
    # # print("Their Total imagination loop duration          :", (time.perf_counter_ns()- start_time_loop)/1000_000, "ms") ### ~ 50ms
    # # torch.cuda.synchronize()
    # # start_time_loop = time.perf_counter_ns()
    # prev_imagined_prior_logits = posteriors_logits.detach().reshape(1, -1, stoch_state_size) ## torch.empty_like(imagined_prior, device=device)
    # prev_recurrent_state = torch.empty_like(recurrent_state, device=device)
    # prev_recurrent_state.copy_(recurrent_state)
    # for i in range(1, cfg.algo.horizon + 1):    ## lopin 15 times
    #     ## TODO: Assumption: currently these values get detached before loss backward, so no WorldModel is trained here 
    #     ##   (so we could combine both _transition calls in imagination for faster inference (one call for actual value, one only for uncertainties))

    #     ## TODO: check is this is currect with imagined_prior, recurrent_state
    #     if episodic_memory is not None:
    #         imagined_prior_logits, recurrent_state, uncertainties = world_model.rssm.imagination(imagined_prior, recurrent_state, actions, \
    #                                                                                       return_logits=True, return_uncertainty = True)
    #     else:
    #         imagined_prior, recurrent_state = world_model.rssm.imagination(imagined_prior, recurrent_state, actions, return_uncertainty = False)
    #         imagined_prior = imagined_prior.view(1, -1, stoch_state_size)   ## after: -> [1, 1024, 1024]
    #     # assert(not torch.equal(prev_imagined_prior, imagined_prior))

    #     ### update treshold based on rolling mean and std ~0.145592ms
    #     if episodic_memory is not None:
    #         read_dream_mean_std[0] = torch.mean(uncertainties) * cfg.episodic_memory.read_exp_mov_avg_alpha + read_dream_mean_std[0] * (1 - cfg.episodic_memory.read_exp_mov_avg_alpha)           
    #         read_dream_mean_std[1] = torch.std(uncertainties) * cfg.episodic_memory.read_exp_mov_avg_alpha + read_dream_mean_std[1] * (1 - cfg.episodic_memory.read_exp_mov_avg_alpha)
    #         lookup_treshold = read_dream_mean_std[0] + read_z * read_dream_mean_std[1]

    #         if cfg.episodic_memory.use_acd:
    #             k: int = cfg.episodic_memory.k_neighbors

    #             uncertainty_mask = uncertainties > lookup_treshold          ## shape [1024]

    #             ## ACDs: [num_uncertainties>_threashold, 1, 1024]
    #             ## TODO: AKTUELL: query = (h_t+1, z_t+1, a_t)
    #             ACDs: torch.tensor = parallel_additive_correction_delta(prev_recurrent_state[:, uncertainty_mask, :], 
    #                                                                 prev_imagined_prior_logits[:, uncertainty_mask, :], 
    #                                                                 actions[:, uncertainty_mask, :], 
    #                                                                 episodic_memory, 
    #                                                                 world_model, 
    #                                                                 k, 
    #                                                                 device=device,
    #                                                                 cfg=cfg)

    #             imagined_prior_logits[:, uncertainty_mask, :] += ACDs

    #             prev_recurrent_state.copy_(recurrent_state)
    #             prev_imagined_prior_logits.copy_(imagined_prior_logits)

    #         imagined_prior = compute_stochastic_state(imagined_prior_logits, discrete=world_model.rssm.discrete, sample=True)
    #         imagined_prior = imagined_prior.view(1, -1, stoch_state_size)

    #     imagined_latent_state = torch.cat((imagined_prior, recurrent_state), -1)
    #     imagined_trajectories[i] = imagined_latent_state
    #     actions = torch.cat(actor(imagined_latent_state.detach())[0], dim=-1)
    #     imagined_actions[i] = actions
    #     # torch.cuda.synchronize()
    #     # print(f"calc parallel ACDs duration: {(time.perf_counter_ns()- start)/1000_000}ms for EM of size: {len(episodic_memory)}")

    # # torch.cuda.synchronize()
    # # print("Total imagination loop duration          :", (time.perf_counter_ns()- start_time_loop)/1000_000, "ms for EM size:", len(episodic_memory))### 
        
    #     # print(f"calc parallel ACDs duration: {(time.perf_counter_ns()- start)/1000_000}ms for EM of size: {len(episodic_memory)}")
    #         # print(f"parallel additive_correction_delta duration: {(time.perf_counter_ns()- start)/1000_000}ms for EM of size: {len(episodic_memory)}")
    #     # print(f"dream lookup threshold is: {lookup_treshold}")
    # ##############################################################################################################################################################################
    # Predict values, rewards and continues
    predicted_values = TwoHotEncodingDistribution(critic(imagined_trajectories), dims=1).mean
    predicted_rewards = TwoHotEncodingDistribution(world_model.reward_model(imagined_trajectories), dims=1).mean
    continues = Independent(BernoulliSafeMode(logits=world_model.continue_model(imagined_trajectories)), 1).mode
    true_continue = (1 - data["terminated"]).flatten().reshape(1, -1, 1)
    continues = torch.cat((true_continue, continues[1:]))

    ##############################################################################################################################################################################
    # # Estimate lambda-values
    # lambda_values = compute_lambda_values(
    #     predicted_rewards[1:],
    #     predicted_values[1:],
    #     continues[1:] * cfg.algo.gamma,
    #     lmbda=cfg.algo.lmbda,
    # )
    ##############################################################################################################################################################################
    # Estimate lambda-values
    lambda_values = compiled_compute_lambda_values(
        predicted_rewards[1:],
        predicted_values[1:],
        continues[1:] * cfg.algo.gamma,
        lmbda=cfg.algo.lmbda,
    )

    # Compute the discounts to multiply the lambda values to
    with torch.no_grad():
        discount = torch.cumprod(continues * cfg.algo.gamma, dim=0) / cfg.algo.gamma

    # Actor optimization step. Eq. 11 from the paper
    # Given the following diagram, with H=3
    # Actions:          [a'0]    [a'1]    [a'2]    a'3
    #                    ^ \      ^ \      ^ \     ^
    #                   /   \    /   \    /   \   /
    #                  /     \  /     \  /     \ /
    # States:       [z0] -> [z'1] -> [z'2] ->  z'3
    # Values:       [v'0]   [v'1]    [v'2]     v'3
    # Lambda-values:        [l'1]    [l'2]    [l'3]
    # Entropies:    [e'0]   [e'1]    [e'2]
    actor_optimizer.zero_grad(set_to_none=True)
    policies: Sequence[Distribution] = actor(imagined_trajectories.detach())[1]

    baseline = predicted_values[:-1]
    offset, invscale = moments(lambda_values, fabric)
    normed_lambda_values = (lambda_values - offset) / invscale
    normed_baseline = (baseline - offset) / invscale
    advantage = normed_lambda_values - normed_baseline
    if is_continuous:
        objective = advantage
    else:
        objective = (
            torch.stack(
                [
                    p.log_prob(imgnd_act.detach()).unsqueeze(-1)[:-1]
                    for p, imgnd_act in zip(policies, torch.split(imagined_actions, actions_dim, dim=-1))
                ],
                dim=-1,
            ).sum(dim=-1)
            * advantage.detach()
        )
    try:
        entropy = cfg.algo.actor.ent_coef * torch.stack([p.entropy() for p in policies], -1).sum(dim=-1)
    except NotImplementedError:
        entropy = torch.zeros_like(objective)
    policy_loss = -torch.mean(discount[:-1].detach() * (objective + entropy.unsqueeze(dim=-1)[:-1]))
    fabric.backward(policy_loss)
    actor_grads = None
    if cfg.algo.actor.clip_gradients is not None and cfg.algo.actor.clip_gradients > 0:
        actor_grads = fabric.clip_gradients(
            module=actor, optimizer=actor_optimizer, max_norm=cfg.algo.actor.clip_gradients, error_if_nonfinite=False
        )
    actor_optimizer.step()

    # Predict the values
    qv = TwoHotEncodingDistribution(critic(imagined_trajectories.detach()[:-1]), dims=1)
    predicted_target_values = TwoHotEncodingDistribution(
        target_critic(imagined_trajectories.detach()[:-1]), dims=1
    ).mean

    # Critic optimization. Eq. 10 in the paper
    critic_optimizer.zero_grad(set_to_none=True)
    value_loss = -qv.log_prob(lambda_values.detach())
    value_loss = value_loss - qv.log_prob(predicted_target_values.detach())
    value_loss = torch.mean(value_loss * discount[:-1].squeeze(-1))

    fabric.backward(value_loss)
    critic_grads = None
    if cfg.algo.critic.clip_gradients is not None and cfg.algo.critic.clip_gradients > 0:
        critic_grads = fabric.clip_gradients(
            module=critic,
            optimizer=critic_optimizer,
            max_norm=cfg.algo.critic.clip_gradients,
            error_if_nonfinite=False,
        )
    critic_optimizer.step()

    # Log metrics
    if aggregator and not aggregator.disabled:
        aggregator.update("Loss/world_model_loss", rec_loss.detach())
        aggregator.update("Loss/observation_loss", observation_loss.detach())
        aggregator.update("Loss/reward_loss", reward_loss.detach())
        aggregator.update("Loss/state_loss", state_loss.detach())
        aggregator.update("Loss/continue_loss", continue_loss.detach())
        aggregator.update("State/kl", kl.mean().detach())
        aggregator.update(
            "State/post_entropy",
            Independent(OneHotCategorical(logits=posteriors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update(
            "State/prior_entropy",
            Independent(OneHotCategorical(logits=priors_logits.detach()), 1).entropy().mean().detach(),
        )
        aggregator.update("Loss/policy_loss", policy_loss.detach())
        aggregator.update("Loss/value_loss", value_loss.detach())
        if world_model_grads:
            aggregator.update("Grads/world_model", world_model_grads.mean().detach())
        if actor_grads:
            aggregator.update("Grads/actor", actor_grads.mean().detach())
        if critic_grads:
            aggregator.update("Grads/critic", critic_grads.mean().detach())

    # Reset everything
    actor_optimizer.zero_grad(set_to_none=True)
    critic_optimizer.zero_grad(set_to_none=True)
    world_optimizer.zero_grad(set_to_none=True)

##### ssshhoouuullddd be correct? ~except maybe the no grad stuff, idk, because we only want to train Dynamics predictor imagination ~Josch
def rehearsal_train(fabric:             Fabric,
                    world_model:        WorldModel,
                    world_optimizer:    Optimizer,
                    
                    cfg: Dict[str, Any],
                    episodic_memory:    EM, 
                    rehearsal_steps:    int     =1,
                    batch_size:         int     =64
                    ) -> None:
    """Runs rehearsal training on EM, updating the sequence and dynamics models of the WM.

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        world_optimizer (Optimizer): the world optimizer.
        cfg (DictConfig): the configs.
        episodic_memory (EM): episodic memory training samples.
        rehearsal_steps (int): number of repeated steps using same samples.
        # aggregator (MetricAggregator, optional): the aggregator to print the metrics.
        batch_size (int): batch size, so number of trajectories used for training in parallel.
    """

    traj_length = episodic_memory.trajectory_length
    recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
    stochastic_size = cfg.algo.world_model.stochastic_size
    discrete_size = cfg.algo.world_model.discrete_size
    device = fabric.device

    # all_recurrents_states:    (1?, num_trajectories, 4096).
    # all_posteriors_logits:    (trajectory_length + 1, num_trajectories, 1024).
    # all_actions:              (trajectory_length + 1, num_trajectories, 1).
    # all_traj_indices: Indices of trajectories actually returned
    all_recurrents_states, all_posteriors_logits, all_actions, all_traj_indices = episodic_memory.get_samples()

    if (all_recurrents_states is None) or (all_posteriors_logits is None) or (all_actions is None) or (all_traj_indices is None): return

    # print("get_samples shape: ", all_recurrents_states.shape, all_posteriors_logits.shape, all_actions.shape) # (1,2,4096)(...)
    # => get_samples shape:  torch.Size([1, 804, 4096]) torch.Size([21, 804, 1024]) torch.Size([21, 804, 18])

    num_trajectories: int = all_recurrents_states.shape[1]
    batch_size = min(batch_size, num_trajectories)
    stoch_state_size = stochastic_size * discrete_size  ## e.g. 32x32

    priors_logits = torch.empty(traj_length + 1, batch_size, stoch_state_size, device=device)   ## +1 always since key value is included
    # print("initial priors_logits shape:", priors_logits.shape)

    ## iterating over each batch of trajectories
    for i in range(0, num_trajectories, batch_size):
        priors_logits = priors_logits.view(traj_length + 1, batch_size, stoch_state_size)

        j = min(batch_size, max(num_trajectories-i, 0)) ### (only relevant for last batch (I mean the calculation))
        if j==0: continue

        ## create batched trajectories
        recurrent_state     = all_recurrents_states[:, i:i+j]
        posteriors_logits   = all_posteriors_logits[:, i:i+j]
        actions             = all_actions[:, i:i+j]

        posteriors = compute_stochastic_state(posteriors_logits, discrete=world_model.rssm.discrete, sample=True) ## torch.empty(traj_length, batch_size, stochastic_size, discrete_size, device=device)
        posteriors = posteriors.view(*posteriors.shape[:-2], -1) ## => (traj_length, batch_size, stochastic_size * discrete_size)
        
        ## setting first z based on initial h
        priors_logits[0, : j], _ = world_model.rssm._transition(recurrent_state)    ## _transition returns: (logits, priors)
        # print("priors_logits shape:", priors_logits.shape, " j:", j)

        ## doing imagination based on initial state
        for k in range(0, traj_length):
            ## ^z, h

            #### THEIR SHAPES: imagined_prior shape: torch.Size([1, 1024, 1024]); recurrent_state shape: torch.Size([1, 1024, 4096]); actions shape: torch.Size([1, 1024, 6])
            if k == 0:
                imagined_prior_logits, recurrent_state, uncertainties = world_model.rssm.imagination(posteriors[k:k+1], recurrent_state, actions[k:k+1], return_logits=True, return_uncertainty=True) ## compute_stochastic_state -> from prior logits to prior
                episodic_memory.uncertainty[all_traj_indices[i:i+j]] = uncertainties    ## Updating uncertainties in EM
            else:
                imagined_prior_logits, recurrent_state = world_model.rssm.imagination(posteriors[k:k+1], recurrent_state, actions[k:k+1], return_logits=True, return_uncertainty=False) ## compute_stochastic_state -> from prior logits to prior

            imagined_prior_logits = imagined_prior_logits.view(1, -1, stoch_state_size)
            priors_logits[k + 1, : j] = imagined_prior_logits

        # Reshape posterior and prior logits to shape [B, T, 32, 32]
        priors_logits = priors_logits.view(*priors_logits.shape[:-1], stochastic_size, discrete_size)
        posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-1], stochastic_size, discrete_size)

        # World model optimization step. Eq. 4 in the paper
        ## Encoder & Decoder will not be optimized, since we did not call them beforehand
        world_optimizer.zero_grad(set_to_none=True)
        ## world_model.encoder.zero_grad(set_to_none=True)
        rec_loss, kl, state_loss = reconstruction_loss_rehearsal(
            priors_logits[:,:j],
            posteriors_logits[:,:j],
            cfg.algo.world_model.kl_dynamic,
            cfg.algo.world_model.kl_representation,
            cfg.algo.world_model.kl_free_nats,
            cfg.algo.world_model.kl_regularizer,
        )
        fabric.backward(rec_loss)
        world_model_grads = None
        if cfg.algo.world_model.clip_gradients is not None and cfg.algo.world_model.clip_gradients > 0:
            world_model_grads = fabric.clip_gradients(
                module=world_model,
                optimizer=world_optimizer,
                max_norm=cfg.algo.world_model.clip_gradients,
                error_if_nonfinite=False,
            )
        world_optimizer.step()

        ### detach shared tensors across loop iteration, because otherwise backward unhappy
        recurrent_state = recurrent_state.detach()
        priors_logits = priors_logits.detach()

    world_optimizer.zero_grad(set_to_none=True)

def update_uncertainties(fabric:             Fabric,
                         world_model:        WorldModel,
                         
                         cfg: Dict[str, Any],
                         episodic_memory:    EM, 
                         batch_size:         int     =64
                        ) -> None:
    """Updates uncertainties in EM by recalculating them from all valid samples.

    Args:
        fabric (Fabric): the fabric instance.
        world_model (_FabricModule): the world model wrapped with Fabric.
        cfg (DictConfig): the configs.
        episodic_memory (EM): episodic memory training samples.
        batch_size (int): batch size, so number of trajectories used for training in parallel.
    """
    ## TODO: reicht das, damit hier nix im torch tree ist?
    with torch.no_grad():
        traj_length = episodic_memory.trajectory_length
        recurrent_state_size = cfg.algo.world_model.recurrent_model.recurrent_state_size
        stochastic_size = cfg.algo.world_model.stochastic_size
        discrete_size = cfg.algo.world_model.discrete_size
        device = fabric.device
        # start_time = time.time()
        all_recurrents_states, all_posteriors_logits, all_actions, all_traj_indices = episodic_memory.get_samples()
        # end_time = time.time()
        # if False:
            # episodic_memory._prune_memory(10)
        # print(f"Get_Samples: {(end_time - start_time)/1000} seconds")

        if all_recurrents_states is None: return
        # print("get_samples types: ", all_recurrents_states.dtype, all_posteriors_logits.dtype, all_actions.dtype)
        # print("get_samples shape: ", all_recurrents_states.shape, all_posteriors_logits.shape, all_actions.shape) # (1,2,4096)(...)
        batch_size = min(batch_size, all_recurrents_states.shape[1])
        stoch_state_size = stochastic_size * discrete_size  ## e.g. 32x32
        # recurrent_state = torch.zeros(1, batch_size, recurrent_state_size, device=device)
        # recurrent_states = torch.empty(traj_length, batch_size, recurrent_state_size, device=device)
        priors_logits = torch.empty(traj_length + 1, batch_size, stoch_state_size, device=device)
        for i in range(0, all_recurrents_states.shape[1], batch_size):
            # print("rehearsal train BATCH_INDEX: ", i)
            priors_logits = priors_logits.view(traj_length + 1, batch_size, stoch_state_size)
            # posteriors_logits = posteriors_logits.view(*posteriors_logits.shape[:-2], stoch_state_size)

            j = min(batch_size, max(all_recurrents_states.shape[1]-i, 0)) ### (only relevant for last batch (i mean the calculation))
            if j==0: continue

            recurrent_state = all_recurrents_states[:, i:i+j]
            posteriors_logits = all_posteriors_logits[:, i:i+j]
            actions = all_actions[:, i:i+j]

            posteriors = compute_stochastic_state(posteriors_logits) ## torch.empty(traj_length, batch_size, stochastic_size, discrete_size, device=device)
            posteriors = posteriors.view(*posteriors.shape[:-2], -1)
            priors_logits[0, : j], _ = world_model.rssm._transition(recurrent_state)

            # print("posteriors_logits shape:", posteriors_logits.shape)
            for k in range(0, traj_length):
                ## ^z, h
                # print("posteriors[k] shape:", posteriors[k].shape)
                # print("recurrent_state shape:", recurrent_state.shape)
                # print("actions[k] shape:", actions[k].shape)
                #### THEIR SHAPES: imagined_prior shape: torch.Size([1, 1024, 1024]); recurrent_state shape: torch.Size([1, 1024, 4096]); actions shape: torch.Size([1, 1024, 6])
                if k == 0:
                    _, recurrent_state, uncertainties = world_model.rssm.imagination(posteriors[k:k+1], recurrent_state, actions[k:k+1], return_logits=True, return_uncertainty=True) ## compute_stochastic_state -> from prior logits to prior
                    ## updating uncertainties in EM
                    episodic_memory.uncertainty[all_traj_indices[i:i+j]] = uncertainties    ## Updating uncertainties

@register_algorithm()
def main(fabric: Fabric, cfg: Dict[str, Any]):
    device = fabric.device
    rank = fabric.global_rank
    world_size = fabric.world_size  ## number of processes (GPUs)?

    ### stores most recent env uncertainty values to calculate avg. and adapt treshold for inserting into EM
    write_z = cfg.episodic_memory.write_std_multiplier_start
    last_N_env_uncertainties = np.zeros((cfg.episodic_memory.write_window_size_N), dtype = np.float32)
    ## rolling mean and std for batched reading while dreaming (Knn lookup)
    read_z = cfg.episodic_memory.read_std_multiplier_start
    read_dream_mean_std = torch.zeros((2), dtype = torch.float32, device=device) ### mean and std for reading from the em (kNN lookup)

    if cfg.checkpoint.resume_from:
        state = fabric.load(cfg.checkpoint.resume_from)

    # These arguments cannot be changed
    cfg.env.frame_stack = -1
    if 2 ** int(np.log2(cfg.env.screen_size)) != cfg.env.screen_size:
        raise ValueError(f"The screen size must be a power of 2, got: {cfg.env.screen_size}")

    # Create Logger. This will create the logger only on the
    # rank-0 process
    logger = get_logger(fabric, cfg)
    if logger and fabric.is_global_zero:
        fabric._loggers = [logger]
        fabric.logger.log_hyperparams(cfg)
    log_dir = get_log_dir(fabric, cfg.root_dir, cfg.run_name)
    fabric.print(f"Log dir: {log_dir}")

    # Environment setup
    vectorized_env = gym.vector.SyncVectorEnv if cfg.env.sync_env else gym.vector.AsyncVectorEnv
    envs = vectorized_env(
        [
            partial(
                RestartOnException,
                make_env(
                    cfg,
                    cfg.seed + rank * cfg.env.num_envs + i,
                    rank * cfg.env.num_envs,
                    log_dir if rank == 0 else None,
                    "train",
                    vector_env_idx=i,
                ),
            )
            for i in range(cfg.env.num_envs)
        ]
    )
    action_space = envs.single_action_space
    observation_space = envs.single_observation_space

    is_continuous = isinstance(action_space, gym.spaces.Box)
    is_multidiscrete = isinstance(action_space, gym.spaces.MultiDiscrete)
    actions_dim = tuple(
        action_space.shape if is_continuous else (action_space.nvec.tolist() if is_multidiscrete else [action_space.n])
    )
    clip_rewards_fn = lambda r: np.tanh(r) if cfg.env.clip_rewards else r
    if not isinstance(observation_space, gym.spaces.Dict):
        raise RuntimeError(f"Unexpected observation type, should be of type Dict, got: {observation_space}")

    if (
        len(set(cfg.algo.cnn_keys.encoder).intersection(set(cfg.algo.cnn_keys.decoder))) == 0
        and len(set(cfg.algo.mlp_keys.encoder).intersection(set(cfg.algo.mlp_keys.decoder))) == 0
    ):
        raise RuntimeError("The CNN keys or the MLP keys of the encoder and decoder must not be disjointed")
    if len(set(cfg.algo.cnn_keys.decoder) - set(cfg.algo.cnn_keys.encoder)) > 0:
        raise RuntimeError(
            "The CNN keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.cnn_keys.decoder))}"
        )
    if len(set(cfg.algo.mlp_keys.decoder) - set(cfg.algo.mlp_keys.encoder)) > 0:
        raise RuntimeError(
            "The MLP keys of the decoder must be contained in the encoder ones. "
            f"Those keys are decoded without being encoded: {list(set(cfg.algo.mlp_keys.decoder))}"
        )
    if cfg.metric.log_level > 0:
        fabric.print("Encoder CNN keys:", cfg.algo.cnn_keys.encoder)
        fabric.print("Encoder MLP keys:", cfg.algo.mlp_keys.encoder)
        fabric.print("Decoder CNN keys:", cfg.algo.cnn_keys.decoder)
        fabric.print("Decoder MLP keys:", cfg.algo.mlp_keys.decoder)
    obs_keys = cfg.algo.cnn_keys.encoder + cfg.algo.mlp_keys.encoder


    # Compile dynamic_learning method
    compiled_dynamic_learning = torch.compile(dynamic_learning, **cfg.algo.compile_dynamic_learning)

    # Compile behaviour_learning method
    compiled_behaviour_learning = torch.compile(behaviour_learning, **cfg.algo.compile_behaviour_learning)

    # Compile compute_lambda_values method
    compiled_compute_lambda_values = torch.compile(compute_lambda_values, **cfg.algo.compile_compute_lambda_values)


    world_model, actor, critic, target_critic, player = build_agent(
        fabric,
        actions_dim,
        is_continuous,
        cfg,
        observation_space,
        state["world_model"] if cfg.checkpoint.resume_from else None,
        state["actor"] if cfg.checkpoint.resume_from else None,
        state["critic"] if cfg.checkpoint.resume_from else None,
        state["target_critic"] if cfg.checkpoint.resume_from else None,
    )

    # Optimizers
    world_optimizer = hydra.utils.instantiate(
        cfg.algo.world_model.optimizer, params=world_model.parameters(), _convert_="all"
    )
    actor_optimizer = hydra.utils.instantiate(cfg.algo.actor.optimizer, params=actor.parameters(), _convert_="all")
    critic_optimizer = hydra.utils.instantiate(cfg.algo.critic.optimizer, params=critic.parameters(), _convert_="all")
    if cfg.checkpoint.resume_from:
        world_optimizer.load_state_dict(state["world_optimizer"])
        actor_optimizer.load_state_dict(state["actor_optimizer"])
        critic_optimizer.load_state_dict(state["critic_optimizer"])
    world_optimizer, actor_optimizer, critic_optimizer = fabric.setup_optimizers(
        world_optimizer, actor_optimizer, critic_optimizer
    )
    moments = Moments(
        cfg.algo.actor.moments.decay,
        cfg.algo.actor.moments.max,
        cfg.algo.actor.moments.percentile.low,
        cfg.algo.actor.moments.percentile.high,
    )
    if cfg.checkpoint.resume_from:
        moments.load_state_dict(state["moments"])

    if fabric.is_global_zero:
        save_configs(cfg, log_dir)

    ################# init EM here #################
    if cfg.episodic_memory.use_episodic_memory:
        episodic_memory: EM = EM(
            trajectory_length=cfg.episodic_memory.trajectory_length,
            uncertainty_threshold=cfg.episodic_memory.uncertainty_threshold,
            max_elements=cfg.episodic_memory.capacity,
            config=cfg,
            k_nn=cfg.episodic_memory.k_neighbors,
            prune_fraction =cfg.episodic_memory.prune_fraction,
            time_to_live = cfg.episodic_memory.time_to_live,
            z_size=cfg.algo.world_model.discrete_size*cfg.algo.world_model.stochastic_size,
            h_size=cfg.algo.world_model.recurrent_model.recurrent_state_size,
            a_size=actions_dim[0], # TODO not sure if shape is correct,
            fabric=fabric
        )
    else:
        # warning that no EM currently used
        warnings.warn("\n!!!!!!!!!! NO EM CURRENTLY USED !!!!!!!!!!\n")

    # Metrics
    aggregator = None
    if not MetricAggregator.disabled:
        aggregator: MetricAggregator = hydra.utils.instantiate(cfg.metric.aggregator, _convert_="all").to(device)

    # Local data
    buffer_size = cfg.buffer.size // int(cfg.env.num_envs * fabric.world_size) if not cfg.dry_run else 2
    rb = EnvIndependentReplayBuffer(
        buffer_size,
        n_envs=cfg.env.num_envs,
        memmap=cfg.buffer.memmap,
        memmap_dir=os.path.join(log_dir, "memmap_buffer", f"rank_{fabric.global_rank}"),
        buffer_cls=SequentialReplayBuffer,
    )
    if cfg.checkpoint.resume_from and cfg.buffer.checkpoint:
        if isinstance(state["rb"], list) and fabric.world_size == len(state["rb"]):
            rb = state["rb"][fabric.global_rank]
        elif isinstance(state["rb"], EnvIndependentReplayBuffer):
            rb = state["rb"]
        else:
            raise RuntimeError(f"Given {len(state['rb'])}, but {fabric.world_size} processes are instantiated")

    # Global variables
    train_step = 0
    last_train = 0
    start_iter = (
        # + 1 because the checkpoint is at the end of the update step
        # (when resuming from a checkpoint, the update at the checkpoint
        # is ended and you have to start with the next one)
        (state["iter_num"] // fabric.world_size) + 1 ### we store it with '* fabric.world_size'
        if cfg.checkpoint.resume_from
        else 1
    )
    ### cumul env steps taken (state["iter_num"] * cfg.env.num_envs) because state["iter_num"] already with '* fabric.world_size'
    policy_step = state["iter_num"] * cfg.env.num_envs if cfg.checkpoint.resume_from else 0
    last_log = state["last_log"] if cfg.checkpoint.resume_from else 0
    last_checkpoint = state["last_checkpoint"] if cfg.checkpoint.resume_from else 0
    ### total parallel envs, with 1 gpu and 1 env = 1
    policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    ### maximum iters, each iter 'policy_steps_per_iter' env steps are 'parallel' taken
    total_iters = int(cfg.algo.total_steps // policy_steps_per_iter) if not cfg.dry_run else 1
    ### in which iter learning starts
    learning_starts = cfg.algo.learning_starts // policy_steps_per_iter if not cfg.dry_run else 0
    prefill_steps = learning_starts - int(learning_starts > 0)
    if cfg.checkpoint.resume_from:
        cfg.algo.per_rank_batch_size = state["batch_size"] // fabric.world_size
        learning_starts += start_iter
        prefill_steps += start_iter

    # Create Ratio class
    ratio = Ratio(cfg.algo.replay_ratio, pretrain_steps=cfg.algo.per_rank_pretrain_steps)
    if cfg.checkpoint.resume_from:
        ratio.load_state_dict(state["ratio"])

    # Warning for log and checkpoint every
    if cfg.metric.log_level > 0 and cfg.metric.log_every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The metric.log_every parameter ({cfg.metric.log_every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the metrics will be logged at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )
    if cfg.checkpoint.every % policy_steps_per_iter != 0:
        warnings.warn(
            f"The checkpoint.every parameter ({cfg.checkpoint.every}) is not a multiple of the "
            f"policy_steps_per_iter value ({policy_steps_per_iter}), so "
            "the checkpoint will be saved at the nearest greater multiple of the "
            "policy_steps_per_iter value."
        )

    # Get the first environment observation and start the optimization
    step_data = {}
    obs = envs.reset(seed=cfg.seed)[0]
    for k in obs_keys:
        step_data[k] = obs[k][np.newaxis]
    step_data["rewards"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["truncated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["terminated"] = np.zeros((1, cfg.env.num_envs, 1))
    step_data["is_first"] = np.ones_like(step_data["terminated"])
    player.init_states()

    ### number of actual optimizer steps, how many "training" steps
    ### cumulative_per_rank_gradient_steps: total training steps
    ### policy_steps:                       total env steps
    cumulative_per_rank_gradient_steps = 0
    
    ### each iter_num, each environment makes a step
    ### policy_step: number of total env interaction: this should be our 100k limit???
    ### with 1 env and 1 gpu: iter_num == policy_step ?
    ### ---------------
    ### policy_steps_per_iter = int(cfg.env.num_envs * fabric.world_size)
    ### so policy_steps_per_iter = total number of parallel env calls, because fabric.world_size = number of gpus??

    last_n_uncertainty_mean = 0.0
    last_n_uncertainty_std  = 0.0

    if cfg.episodic_memory.use_episodic_memory:
        print(f"Fill buffer at max ~{learning_starts} times and then train")
    for iter_num in tqdm(range(start_iter, total_iters + 1)):
        policy_step += policy_steps_per_iter
        # print(f"=== Iteration {iter_num} / {total_iters}, Policy Step: {policy_step} ===")

        ### adjust EM read and write std mulittipliers
        if cfg.episodic_memory.use_episodic_memory:
            if write_z < cfg.episodic_memory.write_std_multiplier_max and \
            policy_step >= cfg.episodic_memory.write_std_inc_start_at_ep:
                write_z = min(cfg.episodic_memory.write_std_multiplier_max, write_z + cfg.episodic_memory.write_std_inc_addend)
            if read_z < cfg.episodic_memory.read_std_multiplier_max and \
            policy_step >= cfg.episodic_memory.read_std_inc_start_at_ep:
                read_z = min(cfg.episodic_memory.read_std_multiplier_max, read_z + cfg.episodic_memory.read_std_inc_addend)

    ### handle environment interaction and replay_buffer filling
    ###-----------------------------------
        # start_time = time.time()
        with torch.inference_mode():
            # Measure environment interaction time: this considers both the model forward
            # to get the action given the observation and the time taken into the environment
            with timer("Time/env_interaction_time", SumMetric, sync_on_compute=False):
                # Sample an action given the observation received by the environment
                if (
                    iter_num <= learning_starts     ### fill replay buffer
                    and cfg.checkpoint.resume_from is None
                    and "minedojo" not in cfg.env.wrapper._target_.lower()
                ):
                    real_actions = actions = np.array(envs.action_space.sample())
                    if not is_continuous:
                        actions = np.concatenate(
                            [
                                F.one_hot(torch.as_tensor(act), act_dim).numpy()
                                for act, act_dim in zip(actions.reshape(len(actions_dim), -1), actions_dim)
                            ],
                            axis=-1,
                        )
                    h, z_logits, uncertainty = None, None, 0.0

                    ## add all samples into EM with high uncertainty to ensure they are stored (they are updated anyways later)
                    if cfg.episodic_memory.use_episodic_memory and cfg.episodic_memory.fill_parallel_to_buffer:
                        torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                        mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                        if len(mask) == 0:
                            mask = None
                        with torch.no_grad():
                            h, z_logits = player.get_hidden_prior(torch_obs, torch.from_numpy(actions).to(device=device).unsqueeze(0)) ## TODO: not performant!!
                        ## shapes z and a:  torch.Size([1, 1, 1024]) torch.Size([1, 1, 6])
                        ## TODO: should we add random actions or predicted actions in EM (currently insert rnd. actions)?
                        uncertainty = torch.tensor([1.0], device=device)
                else:
                    # update_uncertainties(
                    #     fabric          = fabric,
                    #     world_model     = world_model,
                    #     cfg             = cfg,
                    #     episodic_memory = episodic_memory
                    # )

                    torch_obs = prepare_obs(fabric, obs, cnn_keys=cfg.algo.cnn_keys.encoder, num_envs=cfg.env.num_envs)
                    mask = {k: v for k, v in torch_obs.items() if k.startswith("mask")}
                    if len(mask) == 0:
                        mask = None
                    ## player does consist of RSSM und Actor, so lets not only return action but also the rssm stuff
                    ### here and probably during rehearsal training, we need the (1,x) shape, but within the lookup in side the predictor (x,)
                    ### So need to discuss how to store them, (or more, how to restore)
                    if cfg.episodic_memory.use_episodic_memory: 
                        actions, h, z_logits, uncertainty = player.get_actions(torch_obs, mask=mask, return_rssm_stuff=True)
                    else:
                        actions = player.get_actions(torch_obs, mask=mask, return_rssm_stuff=False)
                    real_actions = actions
                    # h, z_logits = h.cpu().numpy(), z_logits.cpu().numpy()
                    actions = torch.cat(actions, -1)
                    if is_continuous:
                        real_actions = torch.stack(real_actions, dim=-1).cpu().numpy()
                    else:
                        real_actions = (
                            torch.stack([real_act.argmax(dim=-1) for real_act in real_actions], dim=-1).cpu().numpy()
                        )
                    
                ### update ems uncertainty treshold
                if cfg.episodic_memory.use_episodic_memory and (z_logits is not None) and (iter_num > learning_starts):
                    last_N_env_uncertainties[iter_num % last_N_env_uncertainties.shape[0]] = uncertainty    ## here add single value
                    last_n_uncertainty_mean = np.mean(last_N_env_uncertainties)
                    last_n_uncertainty_std = np.std(last_N_env_uncertainties)
                    em_read_threshold = last_n_uncertainty_mean + write_z * last_n_uncertainty_std
                    episodic_memory.set_threshold(em_read_threshold)
                    # print(f"uncertainty mean: {last_n_uncertainty_mean} std: {last_n_uncertainty_std}")
                    # print(f"EM read threshold set to: {em_read_threshold}")
                    # print(f"Env Uncertaitny: {uncertainty}")
                    # print(f"EM length: {len(episodic_memory)}")
                
                # em_insert_threshold = np.percentile(last_N_env_uncertainties, cfg.episodic_memory.percentile_treshold)
                # episodic_memory.set_threshold(em_insert_threshold)
                
                ## only relevant for first if (so filling buffer case)
                if type(actions) != np.ndarray:
                    step_data["actions"] = actions.cpu().numpy().reshape((1, cfg.env.num_envs, -1))
                else:
                    step_data["actions"] = actions.reshape((1, cfg.env.num_envs, -1))
                    actions = torch.from_numpy(actions).to(device=device)

                rb.add(step_data, validate_args=cfg.buffer.validate_args)

                ### actual interaction with the environment:
                ### problem: interaction with batch and multi env??? so problem with our em if multiple envs at once because we can only handle one at a time.
                next_obs, rewards, terminated, truncated, infos = envs.step(
                    real_actions.reshape(envs.action_space.shape)
                )
                dones = np.logical_or(terminated, truncated).astype(np.uint8)
                # if h is not None:
                    ## Single action space: Discrete(6)
                    ## Single observation space: Dict('rgb': Box(0, 255, (3, 64, 64), uint8))
                    ## real_actions: [[[2]]]
                    ## SHAPES:
                    ## h: torch.Size([1, 1, 4096])
                    ## z: torch.Size([1, 1, 1024])
                    ## real_actions: (1, 1, 1)
                    ## rewards:      (1,)
                    ## dones:        (1,)
                    # print("Single action space:", envs.single_action_space)
                    # print("Single observation space:", envs.single_observation_space)
                    # print(f"actions: {actions}")
                    # print("SHAPES:")
                    # print(f"  h: {h.shape}")
                    # print(f"  z_logits: {z_logits.shape}")
                    # print(f"  actions: {actions.shape}")
                    # print(f"  rewards:      {rewards.shape}")
                    # print(f"  dones:        {dones.shape}")
                    # print(episodic_memory)
                ## Update Episodic Memory
                # uncertainty = np.array([0.8])
                # if cfg.episodic_memory.enable_rehearsal_training:
                if cfg.episodic_memory.use_episodic_memory:
                    episodic_memory.step(
                        h=h,
                        z=z_logits,
                        a=actions,
                        uncertainty=uncertainty,
                        done=dones
                    )

            step_data["is_first"] = np.zeros_like(step_data["terminated"])
            if "restart_on_exception" in infos:
                for i, agent_roe in enumerate(infos["restart_on_exception"]):
                    if agent_roe and not dones[i]:
                        last_inserted_idx = (rb.buffer[i]._pos - 1) % rb.buffer[i].buffer_size
                        rb.buffer[i]["terminated"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["terminated"][last_inserted_idx]
                        )
                        rb.buffer[i]["truncated"][last_inserted_idx] = np.ones_like(
                            rb.buffer[i]["truncated"][last_inserted_idx]
                        )
                        rb.buffer[i]["is_first"][last_inserted_idx] = np.zeros_like(
                            rb.buffer[i]["is_first"][last_inserted_idx]
                        )
                        step_data["is_first"][i] = np.ones_like(step_data["is_first"][i])

            if cfg.metric.log_level > 0 and "final_info" in infos:
                for i, agent_ep_info in enumerate(infos["final_info"]):
                    if agent_ep_info is not None:
                        ep_rew = agent_ep_info["episode"]["r"]
                        ep_len = agent_ep_info["episode"]["l"]
                        if aggregator and not aggregator.disabled:
                            aggregator.update("Rewards/rew_avg", ep_rew)
                            aggregator.update("Game/ep_len_avg", ep_len)
                        fabric.print(f"Rank-0: policy_step={policy_step}, reward_env_{i}={ep_rew[-1]}")

            # Save the real next observation
            real_next_obs = copy.deepcopy(next_obs)
            if "final_observation" in infos:
                for idx, final_obs in enumerate(infos["final_observation"]):
                    if final_obs is not None:
                        for k, v in final_obs.items():
                            real_next_obs[k][idx] = v

            for k in obs_keys:
                step_data[k] = next_obs[k][np.newaxis]

            # next_obs becomes the new obs
            obs = next_obs

            rewards = rewards.reshape((1, cfg.env.num_envs, -1))
            step_data["terminated"] = terminated.reshape((1, cfg.env.num_envs, -1))
            step_data["truncated"] = truncated.reshape((1, cfg.env.num_envs, -1))
            step_data["rewards"] = clip_rewards_fn(rewards)

            dones_idxes = dones.nonzero()[0].tolist()
            reset_envs = len(dones_idxes)
            if reset_envs > 0:
                reset_data = {}
                for k in obs_keys:
                    reset_data[k] = (real_next_obs[k][dones_idxes])[np.newaxis]
                reset_data["terminated"] = step_data["terminated"][:, dones_idxes]
                reset_data["truncated"] = step_data["truncated"][:, dones_idxes]
                reset_data["actions"] = np.zeros((1, reset_envs, np.sum(actions_dim)))
                reset_data["rewards"] = step_data["rewards"][:, dones_idxes]
                reset_data["is_first"] = np.zeros_like(reset_data["terminated"])
                rb.add(reset_data, dones_idxes, validate_args=cfg.buffer.validate_args)

                # Reset already inserted step data
                step_data["rewards"][:, dones_idxes] = np.zeros_like(reset_data["rewards"])
                step_data["terminated"][:, dones_idxes] = np.zeros_like(step_data["terminated"][:, dones_idxes])
                step_data["truncated"][:, dones_idxes] = np.zeros_like(step_data["truncated"][:, dones_idxes])
                step_data["is_first"][:, dones_idxes] = np.ones_like(step_data["is_first"][:, dones_idxes])
                player.init_states(dones_idxes)
        # end_time = time.time()
        # print(f"env interaction: {(end_time - start_time)/1000} seconds")

    ### handle dreaming interactions and training
    ###-----------------------------------
        # Train the agent
        if iter_num >= learning_starts: ### only with filled replay buffer
            ratio_steps = policy_step - prefill_steps * policy_steps_per_iter   ### ratio_steps = how many env steps happened since training was allowed
            ### conceptually in dreamerv3: gradient_steps = replay_ratio*env_steps, other words: how much training for each env step.
            per_rank_gradient_steps = ratio(ratio_steps / world_size)           # per_rank_gradient_steps: weird logic, but something like: how many gradient steps to run now on each gpu
            # print("per rank gradient step: ", per_rank_gradient_steps, iter_num)
            if per_rank_gradient_steps > 0:
                local_data = rb.sample_tensors(
                    cfg.algo.per_rank_batch_size,
                    sequence_length=cfg.algo.per_rank_sequence_length,
                    n_samples=per_rank_gradient_steps,
                    dtype=None,
                    device=fabric.device,
                    from_numpy=cfg.buffer.from_numpy,
                )
                # start_time = time.time()
                with timer("Time/train_time", SumMetric, sync_on_compute=cfg.metric.sync_on_compute):
                    for i in range(per_rank_gradient_steps):
                        if (
                            cumulative_per_rank_gradient_steps % cfg.algo.critic.per_rank_target_network_update_freq
                            == 0
                        ):
                            tau = 1 if cumulative_per_rank_gradient_steps == 0 else cfg.algo.critic.tau
                            for cp, tcp in zip(critic.module.parameters(), target_critic.parameters()):
                                tcp.data.copy_(tau * cp.data + (1 - tau) * tcp.data)
                        batch = {k: v[i].float() for k, v in local_data.items()}
                        train(
                            fabric,
                            world_model,
                            actor,
                            critic,
                            target_critic,
                            world_optimizer,
                            actor_optimizer,
                            critic_optimizer,
                            batch,
                            aggregator,
                            cfg,
                            is_continuous,
                            actions_dim,
                            moments,
                            compiled_dynamic_learning,
                            compiled_behaviour_learning,
                            compiled_compute_lambda_values,
                            episodic_memory if cfg.episodic_memory.use_episodic_memory else None, 
                            read_dream_mean_std, 
                            read_z
                        )
                        cumulative_per_rank_gradient_steps += 1
                    train_step += world_size
            # end_time = time.time()
            # print(f"training per_rank_gradient_steps:{per_rank_gradient_steps}, seq_len: {cfg.algo.per_rank_sequence_length} times took {(end_time - start_time)/1000} seconds ")
            if cfg.episodic_memory.use_episodic_memory and cfg.episodic_memory.enable_rehearsal_training and iter_num % cfg.episodic_memory.rehearsal_train_every == 0:
                rehearsal_train(
                    fabric          = fabric,
                    world_model     = world_model,
                    world_optimizer = world_optimizer,
                    # aggregator      = aggregator, # add later for metric tracking
                    cfg             = cfg,
                    episodic_memory = episodic_memory
                )
            # print(f"single rehearsal train took: {(end_time - start_time)/1000} seconds ~EM length: {len(episodic_memory)}")

    ### logging
    ###-----------------------------------
        # Log metrics
        if cfg.metric.log_level > 0 and (policy_step - last_log >= cfg.metric.log_every or iter_num == total_iters):
            # Sync distributed metrics
            if aggregator and not aggregator.disabled:
                metrics_dict = aggregator.compute()
                fabric.log_dict(metrics_dict, policy_step)
                aggregator.reset()

            # Log replay ratio
            fabric.log(
                "Params/replay_ratio", cumulative_per_rank_gradient_steps * world_size / policy_step, policy_step
            )

            # Log EM info
            if cfg.episodic_memory.use_episodic_memory: 
                fabric.log("EM/size", len(episodic_memory), policy_step)
                fabric.log("EM/write_z", write_z, policy_step)
                fabric.log("EM/read_z", read_z, policy_step)
                fabric.log("EM/Write_Treshold", (last_n_uncertainty_mean + write_z * last_n_uncertainty_std), policy_step)
                fabric.log("EM/Read_Treshold", (read_dream_mean_std[0].cpu().numpy() + read_z * read_dream_mean_std[1].cpu().numpy()), policy_step) # dont know if correct
                fabric.log("EM/Env_Uncertainty", last_n_uncertainty_mean, policy_step)
                fabric.log("EM/Img_Uncertainty", read_dream_mean_std[0].cpu().numpy(), policy_step) # dont know if correct
                fabric.log("EM/EM_Age_Mean", (torch.mean((episodic_memory.birth_time[:len(episodic_memory)] - episodic_memory.step_counter).type(torch.float32)).cpu().numpy()) * -1, policy_step)
                fabric.log("EM/EM_Age_Std", torch.std((episodic_memory.birth_time[:len(episodic_memory)] - episodic_memory.step_counter).type(torch.float32)).cpu().numpy(), policy_step)
                fabric.log("EM/EM_Uncertainty_Mean", torch.mean(episodic_memory.uncertainty[:len(episodic_memory)]).cpu().numpy(), policy_step)
                fabric.log("EM/EM_Uncertainty_Std", torch.std(episodic_memory.uncertainty[:len(episodic_memory)]).cpu().numpy(), policy_step)
            # Sync distributed timers
            if not timer.disabled:
                timer_metrics = timer.compute()
                if "Time/train_time" in timer_metrics and timer_metrics["Time/train_time"] > 0:
                    fabric.log(
                        "Time/sps_train",
                        (train_step - last_train) / timer_metrics["Time/train_time"],
                        policy_step,
                    )
                if "Time/env_interaction_time" in timer_metrics and timer_metrics["Time/env_interaction_time"] > 0:
                    fabric.log(
                        "Time/sps_env_interaction",
                        ((policy_step - last_log) / world_size * cfg.env.action_repeat)
                        / timer_metrics["Time/env_interaction_time"],
                        policy_step,
                    )
                timer.reset()

            # Reset counters
            last_log = policy_step
            last_train = train_step

        # Checkpoint Model
        if (cfg.checkpoint.every > 0 and policy_step - last_checkpoint >= cfg.checkpoint.every) or (
            iter_num == total_iters and cfg.checkpoint.save_last
        ):
            last_checkpoint = policy_step
            state = {
                "world_model": world_model.state_dict(),
                "actor": actor.state_dict(),
                "critic": critic.state_dict(),
                "target_critic": target_critic.state_dict(),
                "world_optimizer": world_optimizer.state_dict(),
                "actor_optimizer": actor_optimizer.state_dict(),
                "critic_optimizer": critic_optimizer.state_dict(),
                "moments": moments.state_dict(),
                "ratio": ratio.state_dict(),
                "iter_num": iter_num * fabric.world_size,
                "batch_size": cfg.algo.per_rank_batch_size * fabric.world_size,
                "last_log": last_log,
                "last_checkpoint": last_checkpoint,
            }
            ckpt_path = log_dir + f"/checkpoint/ckpt_{policy_step}_{fabric.global_rank}.ckpt"
            fabric.call(
                "on_checkpoint_coupled",
                fabric=fabric,
                ckpt_path=ckpt_path,
                state=state,
                replay_buffer=rb if cfg.buffer.checkpoint else None,
            )

    ## if training finished
    envs.close()
    if fabric.is_global_zero and cfg.algo.run_test:
        test(player, fabric, cfg, log_dir, greedy=False)

    if not cfg.model_manager.disabled and fabric.is_global_zero:
        from sheeprl.algos.dreamer_v1.utils import log_models
        from sheeprl.utils.mlflow import register_model

        models_to_log = {
            "world_model": world_model,
            "actor": actor,
            "critic": critic,
            "target_critic": target_critic,
            "moments": moments,
        }
        register_model(fabric, log_models, cfg, models_to_log)
