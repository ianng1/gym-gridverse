#!/usr/bin/env python
import argparse
import itertools as itt
import time
from typing import Dict
import cv2
import gym
import numpy as np
import pdb
from gym_gridverse.envs.yaml.factory import factory_env_from_yaml
from gym_gridverse.gym import GymEnvironment
from gym_gridverse.outer_env import OuterEnv
from gym_gridverse.representations.observation_representations import (
    make_observation_representation,
)
from gym_gridverse.representations.state_representations import (
    make_state_representation,
)


def make_env(id_or_path: str) -> GymEnvironment:
    """Makes a GV gym environment."""
    try:
        print('Loading using gym.make')
        env = gym.make(id_or_path)

    except gym.error.Error:
        print(f'Environment with id {id_or_path} not found.')
        print('Loading using YAML')
        inner_env = factory_env_from_yaml(id_or_path)
        state_representation = make_state_representation(
            'default',
            inner_env.state_space,
        )
        observation_representation = make_observation_representation(
            'default',
            inner_env.observation_space,
        )
        outer_env = OuterEnv(
            inner_env,
            state_representation=state_representation,
            observation_representation=observation_representation,
        )
        env = GymEnvironment(outer_env)

    else:
        if not isinstance(env, GymEnvironment):
            raise ValueError(
                f'gym id {id_or_path} is not associated with a GridVerse environment'
            )

    return env


def print_compact(data: Dict[str, np.ndarray]):
    """Converts numpy arrays into lists before printing, for more compact output."""
    compact_data = {k: v.tolist() for k, v in data.items()}
    print(compact_data)


def main(args):
    env = make_env(args.id_or_path)
    env.reset()

    spf = 1 / args.fps
    episode = []
    for ei in range(1): #itt.count():
        print(f'# Episode {ei}')
        print()
        
        observation = env.reset()
        episode.append(env.render(mode="rgb_array")[0])

        print('observation:')
        print_compact(observation)
        print()
        time.sleep(spf)
        actions = [0, 0, 0, 1, 0, 1, 0, 1, 0, 1]
        for ti in range(10): #itt.count():
            print(f'episode: {ei}')
            print(f'time: {ti}')
            # pdb.set_trace()

            #0 is down, 1 is up, 2 is right, 3 is left.
            #action = env.action_space.sample()
            action = actions[ti]
            observation, reward, done, _ = env.step(action)
            episode.append(env.render(mode="rgb_array")[0])

            print(f'action: {action}')
            print(f'reward: {reward}')
            print('observation:')
            print_compact(observation)
            print(f'done: {done}')
            print()

            time.sleep(spf)

            if done:
                break
    out = cv2.VideoWriter('output.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 5, (episode[0].shape[1], episode[0].shape[0]), True)
    for i in episode:
        out.write(i)
    out.release()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('id_or_path', help='Gym id or GV YAML file')
    parser.add_argument(
        '--fps', type=float, default=1.0, help='frames per second'
    )
    main(parser.parse_args())
