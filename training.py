import gym_super_mario_bros
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import wrap_mario
from utils import replay_memory, arrange, copy_weights
from Dueling_DQN import Dueling_DQN
import pickle
    

def training_loop(env):
    n_frame = 4
    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    print(COMPLEX_MOVEMENT)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)
    t = 0
    buffer_size = 50000
    update_interval = 50
    print_interval = 10
    score_1st = []
    total_score = 0.0
    loss = 0.0
    num_epochs = 7000

    memory = replay_memory(buffer_size)
    agent = Dueling_DQN(n_frame, env, args)

    for k in range(num_epochs):
        state = arrange(env.reset())
        done = False

        while not done:
            action = agent.select_action(state, env, evaluation=False)
            next_state, reward, done, _ = env.step(action)
            next_state = arrange(next_state)
            total_score += reward
            reward = np.sign(reward) * (np.sqrt(abs(reward) +1) - 1) + 0.001 * reward
            memory.push((state, float(reward), int(action), next_state, int(1 - done)))
            state = next_state 
            stage = env.unwrapped._stage
            if len(memory) > 2000:
                loss += agent.update_parameters(memory)
                t += 1
            if t % update_interval == 0:
                copy_weights(agent.q, agent.q_target)
                agent.save_checkpoint(suffix="", ckpt_path="checkpoints") 

        if k % print_interval == 0:
            print(
                    "Epoch : %d | score : %f | loss : %.2f | stage : %d"

                    %(
                        k,
                        total_score / print_interval,
                        loss / print_interval,
                        stage,
                     )
            )
            score_1st.append(total_score / print_interval)
            total_score = 0
            loss = 0.0
            pickle.dump(score_1st, open("score.p", "wb"))


if __name__ == "__main__":
    args = {}
    training_loop(args)
