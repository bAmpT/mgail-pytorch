import argparse
import numpy as np
import os

from mgail.trainer import Trainer
from mgail.env import Environment
import mgail.buffer as buffer

# Load buffer from different repo
import sys
sys.modules['er'] = buffer
sys.modules['ER'] = buffer


def dispatcher(env):

    trainer = Trainer(env)

    while trainer.itr < env.n_train_iters:

        # Train
        if env.train_mode:
            trainer.train_step()

        # Test
        if trainer.itr % env.test_interval == 0:

            # measure performance
            R = []
            for n in range(env.n_episodes_test):
                R.append(trainer.collect_experience(record=True, vis=env.vis_flag, noise_flag=False, n_steps=1000))

            # update stats
            trainer.reward_mean = sum(R) / len(R)
            trainer.reward_std = np.std(R)

            # print info line
            trainer.print_info_line('full')

            # save snapshot
            if env.train_mode and env.save_models:
                trainer.save_model(dir_name=env.config_dir)

        trainer.itr += 1


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--env_name', type=str, default='InvertedPendulum-v2')
    p.add_argument('--cuda', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    args = p.parse_args()
    
    # load environment
    env = Environment(run_dir=os.path.curdir, env_name=args.env_name)

    # start training
    dispatcher(env=env)
