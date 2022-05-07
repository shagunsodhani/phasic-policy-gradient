import hydra
from mpi4py import MPI
from notifiers import get_notifier
from oc_extras.resolvers import register_new_resolvers
from omegaconf import DictConfig, OmegaConf
from xplogger.logbook import LogBook
from xplogger.utils import serialize_log_to_json

from . import logger, ppg
from . import torch_util as tu
from .envs import get_venv
from .impala_cnn import ImpalaEncoder


def train_fn(
    env_name="coinrun",
    distribution_mode="hard",
    num_levels=200,
    arch="dual",  # 'shared', 'detach', or 'dual'
    # 'shared' = shared policy and value networks
    # 'dual' = separate policy and value networks
    # 'detach' = shared policy and value networks, but with the value function gradient detached during the policy phase to avoid interference
    interacts_total=25_000_000,
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
        num_envs=num_envs,
        env_name=env_name,
        distribution_mode=distribution_mode,
        num_levels=num_levels,
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
def main(config: DictConfig):

    is_debug_job = False
    slurm_id = config.setup.slurm_id
    if slurm_id == "-1":
        # the job is not running on slurm.
        is_debug_job = True
    config_id = config.setup.id
    logbook_config = hydra.utils.instantiate(config.logbook)
    if "mongo" in logbook_config["loggers"] and (
        config_id.startswith("pytest_")
        or config_id in ["sample", "sample_config"]
        or config_id.startswith("test_")
        # or is_debug_job
    ):
        # do not write the job to mongo db.
        print(logbook_config["loggers"].pop("mongo"))
    logbook = LogBook(logbook_config)
    if not is_debug_job:
        zulip = get_notifier("zulip")
        zulip.notify(
            message=f"Starting experiment for config_id: {config_id}. Slurm id is {slurm_id}",
            **config.notifier,
        )
    config_to_write = OmegaConf.to_container(config, resolve=True)
    config_to_write["status"] = "RUNNING"
    config_to_write = OmegaConf.to_container(
        OmegaConf.create(serialize_log_to_json(config_to_write)), resolve=True
    )
    logbook.write_metadata(config_to_write)
    logbook.write_config(config_to_write)

    comm = MPI.COMM_WORLD

    train_fn(
        env_name=config.env_name,
        distribution_mode=config.distribution_mode,
        num_levels=config.num_levels,
        num_envs=config.num_envs,
        n_epoch_pi=config.n_epoch_pi,
        n_epoch_vf=config.n_epoch_vf,
        n_aux_epochs=config.n_aux_epochs,
        n_pi=config.n_pi,
        arch=config.arch,
        comm=comm,
        log_dir=config.setup.save_dir,
    )

    config_to_write["status"] = "COMPLETED"
    logbook.write_metadata(config_to_write)

    if not is_debug_job:
        zulip.notify(
            message=f"Completed experiment for config_id: {config_id}. Slurm id is {slurm_id}",
            **config.notifier,
        )


if __name__ == "__main__":
    register_new_resolvers()
    main()
