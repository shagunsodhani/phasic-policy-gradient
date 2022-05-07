import argparse

import hydra
from mpi4py import MPI
from oc_extras.resolvers import register_new_resolvers
from omegaconf import DictConfig, OmegaConf

from . import logger, ppg
from . import torch_util as tu
from .envs import get_venv
from .impala_cnn import ImpalaEncoder


def train_fn(
    env_name="coinrun",
    distribution_mode="hard",
    arch="dual",  # 'shared', 'detach', or 'dual'
    # 'shared' = shared policy and value networks
    # 'dual' = separate policy and value networks
    # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
    interacts_total=100_000_000,
    num_envs=64,
    n_epoch_pi=1,
    n_epoch_vf=1,
    gamma=0.999,
    aux_lr=5e-4,
    lr=5e-4,
    nminibatch=8,
    aux_mbsize=4,
    clip_param=0.2,
    kl_penalty=0.0,
    n_aux_epochs=6,
    n_pi=32,
    beta_clone=1.0,
    vf_true_weight=1.0,
    log_dir="/tmp/ppg",
    comm=None,
):
    if comm is None:
        comm = MPI.COMM_WORLD
    tu.setup_dist(comm=comm)
    tu.register_distributions_for_tree_util()

    if log_dir is not None:
        format_strs = ["json", "stdout"] if comm.Get_rank() == 0 else []
        logger.configure(comm=comm, dir=log_dir, format_strs=format_strs)

    venv = get_venv(
        num_envs=num_envs, env_name=env_name, distribution_mode=distribution_mode
    )

    enc_fn = lambda obtype: ImpalaEncoder(
        obtype.shape,
        outsize=256,
        chans=(16, 32, 32),
    )
    model = ppg.PhasicValueModel(venv.ob_space, venv.ac_space, enc_fn, arch=arch)

    model.to(tu.dev())
    logger.log(tu.format_model(model))
    tu.sync_params(model.parameters())

    name2coef = {"pol_distance": beta_clone, "vf_true": vf_true_weight}

    ppg.learn(
        venv=venv,
        model=model,
        interacts_total=interacts_total,
        ppo_hps=dict(
            lr=lr,
            γ=gamma,
            λ=0.95,
            nminibatch=nminibatch,
            n_epoch_vf=n_epoch_vf,
            n_epoch_pi=n_epoch_pi,
            clip_param=clip_param,
            kl_penalty=kl_penalty,
            log_save_opts={"save_mode": "last"},
        ),
        aux_lr=aux_lr,
        aux_mbsize=aux_mbsize,
        n_aux_epochs=n_aux_epochs,
        n_pi=n_pi,
        name2coef=name2coef,
        comm=comm,
    )


@hydra.main(
    config_path="/private/home/sodhani/projects/phasic-policy-gradient/conf",
    config_name="config",
)
def main(cfg: DictConfig):

    comm = MPI.COMM_WORLD

    train_fn(
        env_name=cfg.env_name,
        num_envs=cfg.num_envs,
        n_epoch_pi=cfg.n_epoch_pi,
        n_epoch_vf=cfg.n_epoch_vf,
        n_aux_epochs=cfg.n_aux_epochs,
        n_pi=cfg.n_pi,
        arch=cfg.arch,
        comm=comm,
        log_dir=cfg.setup.save_dir,
    )


if __name__ == "__main__":
    register_new_resolvers()
    main()
