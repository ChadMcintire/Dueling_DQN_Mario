import argparse

from training import training_loop
#from execute_trained_model import run

from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from wrappers import wrap_mario
import gym_super_mario_bros

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", default=0.0001, help="model learing rate (default=.0001)")
    parser.add_argument("--gamma", default=0.99, help="discount factor (default=.99)")
    parser.add_argument("--batch_size", default=256, type=int, help="Batch size for both actor and critic, (default is 256)")
    parser.add_argument("--epsilon", default=0.001, help="The probability of choosing a random action (default=.001)")
    parser.add_argument("--num_epochs", default=int(7e3), type=int, help="The number of episode before the end (default=7000)")
    parser.add_argument("--buffer_size", default=int(5e4), type=int, help="The number of episode before the end (default=50000)")
    parser.add_argument("--update_interval", default=50, type=int, help="How often weights are copied to the target network (default=50)")
    parser.add_argument("--print_interval", default=10, type=int, help="How often we write to the summary writer or print to screen(default=50)")
    parser.add_argument("--n_frame", default=4, type=int, help="Number of frames in a state (default=4)")
    parser.add_argument('--train', dest="train", action='store_true', help='train a new model')
    parser.add_argument('--no-train', dest="train", action='store_false', help='load a previously trained model and run it')
    parser.set_defaults(train=True)

    args = parser.parse_args()

    env = gym_super_mario_bros.make("SuperMarioBros-v0")
    print(COMPLEX_MOVEMENT)
    env = JoypadSpace(env, COMPLEX_MOVEMENT)
    env = wrap_mario(env)


    if args.train:
        training_loop(env, args)
    else:
        run(env, args)
