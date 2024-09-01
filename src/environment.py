import cv2
import numpy as np
import torch.multiprocessing as mp

# Gym is an OpenAI toolkit for RL
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# # NES Emulator for OpenAI Gym
from nes_py.wrappers import JoypadSpace

# # Super Mario environment for OpenAI Gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

class SkipFrame(gym.Wrapper):
    def __init__(self, env, skip):
        """Return only every `skip`-th frame"""
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        """Repeat action, and sum reward"""
        total_reward = 0.0
        for i in range(self._skip):
            # Accumulate reward and repeat the same action
            obs, reward, done, trunk, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return obs, total_reward, done, trunk, info

class GrayScaleResizeObservation(gym.ObservationWrapper):
    def __init__(self, env, shape):
        super().__init__(env)
        if isinstance(shape, int):
            self.shape = (shape, shape)
        else:
            self.shape = tuple(shape)

        obs_shape = self.shape + self.observation_space.shape[2:]
        self.observation_space = Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def observation(self, observation):
        self.current_state = observation
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        observation = cv2.resize(observation, self.shape, interpolation=cv2.INTER_AREA)
        observation = observation.astype(np.uint8)#.reshape(-1, observation.shape[0], observation.shape[1])
        return observation

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        super(NoopResetEnv, self).__init__(env)
        self.noop_max = noop_max

    def reset(self, **kwargs):
        """Do no-op action for a number of steps in [1, noop_max]."""
        obs = self.env.reset(**kwargs)
        noops = np.random.randint(0, self.noop_max, (1, ))[0]
        for _ in range(noops):
            action = self.env.action_space.sample()
            obs, _, done, _, _ = self.env.step(action)
            if done:
                obs = self.env.reset(**kwargs)
        return obs

    def step(self, ac):
        obs, reward, done, trunk, info = self.env.step(ac)
        return obs, reward, done, trunk, info

class CustomRewardAndDoneEnv(gym.Wrapper):
    def __init__(self, env=None, world=1, stage=1):
        super(CustomRewardAndDoneEnv, self).__init__(env)
        self.current_score = 0
        self.current_x = 0
        self.old_x = -1
        self.current_x_count = 0
        self.max_x = 0
        self.world = world
        self.stage = stage
        if self.world == 8 and self.stage == 4:
            self.sea_map = False

    def reset(self, **kwargs):
        self.current_score = 0
        self.current_x = 0
        self.old_x = -1
        self.current_x_count = 0
        self.max_x = 0
        if self.world == 8 and self.stage == 4:
            self.sea_map = False
        return self.env.reset(**kwargs)

    def step(self, action):
        state, reward, done, trunc, info = self.env.step(action)

        if (info['x_pos'] - self.current_x) == 0:
            self.current_x_count += 1
        else:
            self.current_x_count = 0
        if info["flag_get"]:
            reward += 50
            done = True
        if done and info["flag_get"] == False and info["time"] != 0:
            reward -= 50
            done = True
        self.current_x = info["x_pos"]

        if self.world == 7 and self.stage == 4:
            if (506 <= info["x_pos"] <= 832 and info["y_pos"] > 127) or (
                    832 < info["x_pos"] <= 1064 and info["y_pos"] < 80) or (
                    1113 < info["x_pos"] <= 1464 and info["y_pos"] < 191) or (
                    1579 < info["x_pos"] <= 1943 and info["y_pos"] < 191) or (
                    1946 < info["x_pos"] <= 1964 and info["y_pos"] >= 191) or (
                    1984 < info["x_pos"] <= 2060 and (info["y_pos"] >= 191 or info["y_pos"] < 127)) or (
                    2114 < info["x_pos"] < 2440 and info["y_pos"] < 191):
                reward -= 50
                done = True
            if done == False and info["x_pos"] < self.max_x - 100:
                done = True
        if self.world == 4 and self.stage == 4:
            if (info["x_pos"] <= 1500 and info["y_pos"] < 127) or (
                    1588 <= info["x_pos"] < 2380 and info["y_pos"] >= 127):
                reward -= 50
                done = True
            if done == False and info["x_pos"] < self.max_x - 100:
                done = True
            if done == False:
                reward -= 0.1
        if self.world == 4 and self.stage == 2 and done == False and info['y_pos'] >= 255:
            reward -= 50
        if self.world == 8 and self.stage == 4:
            if info["x_pos"] > 2440 and info["x_pos"] <= 2500:
                done = True
                reward -= 100
            if info["x_pos"] >= 3675 and info["x_pos"] <= 3700:
                done = True
                reward -= 50

            if info["x_pos"] < self.max_x - 200:
                if self.max_x >= 1240 and self.max_x <= 1310: #solved bug because x_pos duplicated
                    if info["x_pos"] >= 320:
                        done = True
                        reward -= 50

            if info["x_pos"] < self.old_x - 200:
                if info["x_pos"] >= 312-5 and info["x_pos"] <= 312+5:
                    done = True
                    reward -= 50
                elif info["x_pos"] >= 56-5 and info["x_pos"] <= 56+5 and self.max_x > 3645 and self.sea_map == False:
                    reward += 50
                    self.sea_map = True
            if info["x_pos"] > self.max_x + 100:
                reward += 50
            if done == False:
                reward -= 0.1
        self.max_x = max(self.max_x, self.current_x)
        self.current_score = info["score"]
        self.old_x = self.current_x

        return state, reward / 10., done, trunc, info
    
def create_env(world, stage, action_type, test=False):
    if gym.__version__ < '0.26':
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0", new_step_api=True)
    else:
        env = gym_super_mario_bros.make(f"SuperMarioBros-{world}-{stage}-v0", render_mode='rgb', apply_api_compatibility=True)

    if action_type == "right":
        action_type = RIGHT_ONLY
    elif action_type == "simple":
        action_type = SIMPLE_MOVEMENT
    else:
        action_type = COMPLEX_MOVEMENT

    env = JoypadSpace(env, action_type)

    if test == False:
        env = NoopResetEnv(env)
    env = SkipFrame(env, skip=4)
    env = CustomRewardAndDoneEnv(env, world, stage)
    env = GrayScaleResizeObservation(env, shape=84)
    if gym.__version__ < '0.26':
        env = FrameStack(env, num_stack=4, new_step_api=True)
    else:
        env = FrameStack(env, num_stack=4)
    return env

class MultipleEnvironments:
    def __init__(self, world, stage, action_type, num_envs):
        self.agent_conns, self.env_conns = zip(*[mp.Pipe(duplex=True) for _ in range(num_envs)])
        self.envs = [create_env(world, stage, action_type) for _ in range(num_envs)]

        for index in range(num_envs):
            process = mp.Process(target=self.run, args=(index,))
            process.start()
            self.env_conns[index].close()

    def run(self, index):
        self.agent_conns[index].close()
        while True:
            request, action = self.env_conns[index].recv()
            if request == "step":
                next_state, reward, done, trunc, info = self.envs[index].step(action)
                if done:
                    next_state = self.envs[index].reset()
                self.env_conns[index].send((next_state, reward, done, trunc, info))
            elif request == "reset":
                self.env_conns[index].send(self.envs[index].reset())
            else:
                raise NotImplementedError

    def step(self, actions):
        [agent_conn.send(("step", act)) for agent_conn, act in zip(self.agent_conns, actions)]
        next_states, rewards, dones, truncs, infos = zip(*[agent_conn.recv() for agent_conn in self.agent_conns])
        return next_states, rewards, dones, truncs, infos

    def reset(self):
        [agent_conn.send(("reset", None)) for agent_conn in self.agent_conns]
        states = [agent_conn.recv() for agent_conn in self.agent_conns]
        return states