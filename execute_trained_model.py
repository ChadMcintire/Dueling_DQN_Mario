from Dueling_DQN import Dueling_DQN
from utils import arrange
import time

import torch 
def run(env, args):
    print("beginning runs")
    n_frame = args.n_frame
    agent = Dueling_DQN(n_frame, env, args)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    agent = Dueling_DQN(n_frame, env, args)

    checkpoint = args.ch_pt
    
    #agent.load_checkpoint("./checkpoints.pth", evaluate=True)
    agent.load_checkpoint(checkpoint, evaluate=True)


    #i = 0
    while True:
        total_score = 0.0
        done = False
        s = state = arrange(env.reset())
        while not done:
            env.render()
            action = agent.select_action(state, env, evaluation=False)
            next_state, reward, done, _ = env.step(action)
            next_state = arrange(next_state)
            total_score += reward
            state = next_state
            time.sleep(0.01)

        stage = env.unwrapped._stage
        print("Total score : %f | stage : %d" % (total_score, stage))
