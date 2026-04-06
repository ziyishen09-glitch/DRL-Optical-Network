from argparse import Namespace
from PPO.PPO_train_online import plot_reward_progress

if __name__ == "__main__":
    args = Namespace(log_dir="tmp/rwa_ppo", plot_every=1)
    plot_reward_progress(args, args.plot_every)
