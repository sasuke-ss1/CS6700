from utils import *
seed_everything(42)
from model import *
from argparse import ArgumentParser
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import MSELoss
from collections import deque
import yaml
import wandb
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # Select device to train on

def train(args: ArgumentParser, env: Env, use_max = None) -> list:
    '''
    Helper function to generate the comparision plots, by using rewards
    from 5 independent runs.
    '''

    gamma = args.gamma
    explore = args.explore
    init_eps = args.init_eps
    fin_eps = args.fin_eps
    curr_eps = init_eps
    buffer_size = args.buffer_size
    batch_size = args.batch_size
    target_update_freq = args.target_update_freq
    epochs = args.epochs
    max_timesteps = args.max_time_steps
    warm_up_steps = args.warm_up_steps
    if use_max != None:
        use_max = use_max
    else:
        use_max = args.use_max
    hidden_size1 = args.hidden_size1
    hidden_size2 = args.hidden_size2
    n_state = env.observation_space.shape[0]
    n_action = env.action_space.n
    memory = Buffer(buffer_size) 
    QNet = DDQN(n_state, n_action, [hidden_size1, hidden_size2], args.activation, use_max=use_max).to(device)
    targetQNet = DDQN(n_state, n_action, [hidden_size1, hidden_size2], args.activation, use_max=use_max).to(device)
    targetQNet.load_state_dict(QNet.state_dict())
    optimizer = Adam(QNet.parameters(), lr=args.learning_rate)
    lossfn = MSELoss()

    print('Warming up')
    
    # Load the buffer with some experience
    while len(memory) < warm_up_steps:
        state = env.reset()[0]
        while True:
            p = np.random.rand()
            if p < init_eps:
                action = np.random.randint(0, n_action)
            else:
                state_ = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = QNet.select_action(state_)

            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, next_state, reward, done))
            if done or len(memory) >= warm_up_steps: 
                break
            state = next_state

    print('Warming up finished')

    loop_obj = tqdm(range(epochs))
    episodic_rewards = []
    movAvg = deque(maxlen=100)
    for epoch in loop_obj:
        losses = []
        state = env.reset()[0]
        episode_rewards = []
    
        for steps in range(max_timesteps):
            optimizer.zero_grad()
            p = np.random.rand()
            
            # Select action using epsilon greedy policy
            if p < curr_eps:
                action = np.random.randint(0, n_action)
            else:
                state_ = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                action = QNet.select_action(state_)

            next_state, reward, done, _, _ = env.step(action)
            memory.append((state, action, next_state, reward, done))
            episode_rewards.append(reward)
            
            if steps % target_update_freq == 0:
                targetQNet.load_state_dict(QNet.state_dict()) # Copy the policy Q network params to target Q network

            # Update the policy Q network
            batch_samples = memory.sample(batch_size, False)
            batch_state, batch_action, batch_next_state, batch_reward, batch_done =  zip(*batch_samples)

            batch_state = torch.tensor(batch_state, dtype=torch.float32).to(device)
            batch_next_state = torch.tensor(batch_next_state, dtype=torch.float32).to(device)
            batch_action = torch.tensor(batch_action, dtype=torch.float32).unsqueeze(-1).to(device)
            batch_reward = torch.tensor(batch_reward, dtype=torch.float32).unsqueeze(-1).to(device)
            batch_done = torch.tensor(batch_done, dtype=torch.float32).unsqueeze(-1).to(device)
            
            with torch.no_grad():
                nxt = QNet(batch_next_state)
                tgt_nxt = targetQNet(batch_next_state)
                max_action = torch.argmax(nxt, dim=1, keepdim=True)
                tgt = batch_reward + (1 - batch_done) * gamma * tgt_nxt.gather(1, max_action.long())

            loss = lossfn(QNet(batch_state).gather(1, batch_action.long()), tgt)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

            if curr_eps > fin_eps:
                curr_eps -= (init_eps - fin_eps) / explore

            if done:
                break
            state = next_state
        movAvg.append(sum(episode_rewards))
        episodic_rewards.append(sum(episode_rewards))
        loop_obj.set_postfix_str(f'Loss: {sum(losses)/len(losses)}, Reward: {sum(movAvg)/100}')

    return episodic_rewards


def train_wb():
    '''
    Wandb hyperparameter search helper function
    '''
    run = wandb.init() # Initialize the run
    config = wandb.config # Get the config

    if config.use_max == False:
        name = 'AVG'
    else:
        name = 'MAX'

    wandb.run.name = f'DDQN_{name}_{config.gamma}_{config.batch_size}_[{config.hidden_size1},{config.hidden_size2}]_{config.buffer_size}_[{config.init_eps},{config.fin_eps}]'

    num_expts = 5
    episodic_reward_avgs = []
    for i in range(num_expts):
        episodic_rewards = train(config,env,use_max=config.use_max)
        episodic_reward_avgs.append(episodic_rewards)

    episodic_reward_avgs = np.mean(np.array(episodic_reward_avgs),axis = 0)

    for rewards in episodic_reward_avgs[::50]:
        wandb.log({
            'Episodic reward': rewards
        })


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--wandb_project', '-wp', default='RL-A2-DDQN-Acrobot', type=str, help="The wandb project name where run will be stored")
    parser.add_argument('--wandb', '-wb', default=False, type=bool, help="Run WandB")
    parser.add_argument('--learning_rate', '-lr', default=1e-3, type=float, help='Learning rate for the Q network')
    parser.add_argument('--gamma', '-g', default=0.99, type=float, help="Discount Factor")
    parser.add_argument('--explore', '-ex', default=20000, type=int, help="Exploration Steps")
    parser.add_argument('--init_eps', '-ei', default=1e-1, type=float, help="Initial Learning Rate")
    parser.add_argument('--fin_eps', '-ef', default=1e-4, type=float, help="Final Learning Rate")
    parser.add_argument('--buffer_size', '-bs', default=50000, type=int, help="Buffer Size")
    parser.add_argument('--batch_size', '-b', default=16, type=int, help="Batch Size")
    parser.add_argument('--target_update_freq', '-t', default=4, type=int, help="Target Update Frequency")
    parser.add_argument('--epochs', '-e', default=1000, type=int, help="Epochs")
    parser.add_argument('--max_time_steps', '-mt', default=200, type=int, help="Max Time Steps in an Episode")
    parser.add_argument('--warm_up_steps', '-wu', default=250, type=int, help="Warm up steps")
    parser.add_argument('--use_max', '-um', default=False, type=bool, help="Warm up steps")
    parser.add_argument('--hidden_size1', '-h1', default=64, type=int, help="Hidden Size 1")
    parser.add_argument('--hidden_size2', '-h2', default=256, type=int, help="Hidden Size 2")
    parser.add_argument('--environment', '-env', default=0, type=int, help="Select the environment 0: CartPole-v1, 1: Acrobot-v2")
    parser.add_argument('--fig_name', '-fn', default='abc', type=str, help='Saving name of the plot.')
    parser.add_argument('--activation', '-act', default='ReLU', type=str, help="Activation function used in the model")
    args = parser.parse_args()
        
    env = build_env('Acrobot-v1' if args.environment else 'CartPole-v1') # Build environment
    if args.wandb == False:
        num_expts = 5
        episodic_rewards_list_AVG = []
        episodic_rewards_list_MAX = []
        for i in range(num_expts):
            print(f"Experiment {i+1}")
            episodic_rewards = train(args,env,use_max = False)
            episodic_rewards_list_AVG.append(episodic_rewards[::10])
            episodic_rewards = train(args,env,use_max = True)
            episodic_rewards_list_MAX.append(episodic_rewards[::10])
        episodic_rewards_AVG_avg = np.mean(np.array(episodic_rewards_list_AVG), axis=0)
        episodic_rewards_AVG_std = np.std(np.array(episodic_rewards_list_AVG), axis=0)
        episodic_rewards_MAX_avg = np.mean(np.array(episodic_rewards_list_MAX), axis=0)
        episodic_rewards_MAX_std = np.std(np.array(episodic_rewards_list_MAX), axis=0)
        plt.figure(figsize = (10,7))
        plt.plot(episodic_rewards_AVG_avg, label = 'Average')
        plt.fill_between(range(len(episodic_rewards_AVG_avg)),episodic_rewards_AVG_avg - episodic_rewards_AVG_std,episodic_rewards_AVG_avg + episodic_rewards_AVG_std, color='lightblue', alpha=0.5, label='Mean ± Std Dev')
        plt.plot(episodic_rewards_MAX_avg, label = 'Max')
        plt.fill_between(range(len(episodic_rewards_MAX_avg)),episodic_rewards_MAX_avg - episodic_rewards_MAX_std,episodic_rewards_MAX_avg + episodic_rewards_MAX_std, color='orange', alpha=0.5, label='Mean ± Std Dev')
        plt.legend()
        plt.xlabel('Episode')
        plt.ylabel('Episodic rewards')
        plt.savefig(args.fig_name + '.png')
    else:
        wandb.login(key="ffbdeb8b8eb61fe76925bb00113546a4c1d0581e")
        with open("./sweep_DDQN.yml", "r") as f:
            sweep_config = yaml.safe_load(f)

        sweep_id = wandb.sweep(sweep_config, project=args.wandb_project)
        sweep_agent = wandb.agent(sweep_id, function=train_wb)