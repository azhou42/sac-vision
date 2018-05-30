import argparse

from rllab.envs.normalized_env import normalize
from rllab.misc.instrument import VariantGenerator
from rllab import config

from sac.algos.ddpg import DDPG 
from sac.envs.gym_env import GymEnv
from sac.misc.instrument import run_sac_experiment
from sac.misc.utils import timestamp
from sac.misc.sampler import SimpleSampler
from sac.policies.gmm import GMMPolicy
from sac.policies.uniform_policy import UniformPolicy
from sac.replay_buffers import SimpleReplayBuffer
from sac.value_functions import NNQFunction, NNVFunction
import numpy as np

config.DOCKER_IMAGE = "haarnoja/sac"  # needs psutils
config.AWS_IMAGE_ID = "ami-a3a8b3da"  # with docker already pulled

COMMON_PARAMS = {
    "seed": [2 + 10*i for i in range(10)],
    "lr": [3E-4],
    "discount": 0.99,
    "tau": 1,
    "target_update_freq": 1000,
    "K": 1,
    "layer_size": 256,
    "batch_size": 256, # 100?
    "max_pool_size": 1e6, # [1e3, 1e4, 1e5],
    "learn_alpha": False,
    "n_train_repeat": 1,
    "n_random_steps": 1000,
    "epoch_length": 1000,
    "freeze_pool_after_full": False, # make sure this is false for real runs
    "reparameterize": True,
    "snapshot_mode": 'gap',
    "snapshot_gap": 100,
    "sync_pkl": True,
}


ENV_PARAMS = {
    'swimmer': { # 2 DoF
        'prefix': 'swimmer',
        'env_name': 'swimmer-rllab',
        'max_path_length': 1000,
        'n_epochs': 2000,
        'scale_reward': 100,
    },
    'hopper': { # 3 DoF
        'prefix': 'hopper',
        'env_name': 'Hopper-v1',
        'max_path_length': 1000,
        'n_epochs': 3000,
        'scale_reward': [1,3],
    },
    'half-cheetah': { # 6 DoF
        'prefix': 'half-cheetah',
        'env_name': 'HalfCheetah-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        "n_random_steps": 10000,
        'scale_reward': [1], # [np.exp(r) for r in np.linspace(np.log(0.1), np.log(100), num=40)], # 1, # scale rewards from 0.1 to 100
    },
    'walker': { # 6 DoF
        'prefix': 'walker',
        'env_name': 'Walker2d-v1',
        'max_path_length': 1000,
        'n_epochs': 1000,
        "n_random_steps": 1000,
        'scale_reward': [1],
    },
    'ant': { # 8 DoF
        'prefix': 'ant',
        'env_name': 'Ant-v1',
        'max_path_length': 1000,
        'n_epochs': 2000,
        "n_random_steps": 10000,
        'scale_reward': [1], # [np.exp(r) for r in np.linspace(np.log(0.1), np.log(100), num=40)],
    },
    'humanoid': { # 17 DoF
        'prefix': 'humanoid',
        'env_name': 'Humanoid-v1', # switch to gym humanoid
        'max_path_length': 1000,
        'n_epochs': 10000,
        "n_random_steps": 1000,
        'scale_reward': [1], # [5,10,20,40],
    },
}
DEFAULT_ENV = 'swimmer'
AVAILABLE_ENVS = list(ENV_PARAMS.keys())

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--env',
                        type=str,
                        choices=AVAILABLE_ENVS,
                        default='swimmer')
    parser.add_argument('--exp_name',type=str, default=timestamp())
    parser.add_argument('--mode', type=str, default='local')
    parser.add_argument('--log_dir', type=str, default=None)
    args = parser.parse_args()

    return args


def get_variants(args):
    env_params = ENV_PARAMS[args.env]
    params = COMMON_PARAMS
    params.update(env_params)

    vg = VariantGenerator()
    for key, val in params.items():
        if isinstance(val, list):
            vg.add(key, val)
        else:
            vg.add(key, [val])

    return vg


def run_experiment(variant):
    if variant['env_name'] == 'humanoid-rllab':
        from rllab.envs.mujoco.humanoid_env import HumanoidEnv
        env = normalize(HumanoidEnv())
    elif variant['env_name'] == 'swimmer-rllab':
        from rllab.envs.mujoco.swimmer_env import SwimmerEnv
        env = normalize(SwimmerEnv())
    else:
        env = normalize(GymEnv(variant['env_name']))

    pool = SimpleReplayBuffer(
        env_spec=env.spec,
        freeze_after_full=variant['freeze_pool_after_full'],
        max_replay_buffer_size=variant['max_pool_size'],
    )

    sampler = SimpleSampler(
        max_path_length=variant['max_path_length'],
        min_pool_size=variant['max_path_length'],
        batch_size=variant['batch_size']
    )

    base_kwargs = dict(
        sampler=sampler,
        epoch_length=variant['epoch_length'],
        n_epochs=variant['n_epochs'],
        n_random_steps=variant['n_random_steps'],
        n_train_repeat=variant['n_train_repeat'],
        eval_render=False,
        eval_n_episodes=1,
        eval_deterministic=True,
    )

    M = variant['layer_size']
    qf1 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        name='qf1',
    )
    
    qf2 = NNQFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
        name='qf2',
    )

    vf = NNVFunction(
        env_spec=env.spec,
        hidden_layer_sizes=[M, M],
    )

    uniform_policy = UniformPolicy(env_spec=env.spec)

    policy = GMMPolicy(
        env_spec=env.spec,
        K=variant['K'],
        hidden_layer_sizes=[M, M],
        qf=qf1,
        reparameterize=variant['reparameterize'],
        reg=0.001,
    )
    

    algorithm = DDPG(
        base_kwargs=base_kwargs,
        env=env,
        policy=policy,
        uniform_policy=uniform_policy,
        pool=pool,
        qf1=qf1,
        qf2=qf2,

        lr=variant['lr'],
        scale_reward=variant['scale_reward'],
        discount=variant['discount'],
        tau=variant['tau'],
        target_update_freq=variant['target_update_freq'],

        save_full_state=False,
    )

    algorithm.train()


def launch_experiments(variant_generator):
    variants = variant_generator.variants()

    for i, variant in enumerate(variants):
        print('Launching {} experiments.'.format(len(variants)))
        run_sac_experiment(
            run_experiment,
            mode=args.mode,
            variant=variant,
            exp_prefix='nips' + '/' + variant['prefix'] + '/' + args.exp_name,
            exp_name=variant['prefix'] + '-' + args.exp_name + '-' + str(i).zfill(2),
            n_parallel=1,
            seed=variant['seed'],
            terminate_machine=True,
            log_dir=args.log_dir,
            snapshot_mode=variant['snapshot_mode'],
            snapshot_gap=variant['snapshot_gap'],
            sync_s3_pkl=variant['sync_pkl'],
        )

if __name__ == '__main__':
    args = parse_args()
    variant_generator = get_variants(args)
    launch_experiments(variant_generator)
