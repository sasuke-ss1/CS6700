from utils import *
seed_everything(42)
from model import *
from argparse import ArgumentParser
from torch.optim import Adam
from tqdm import tqdm
from torch.nn import MSELoss
from collections import deque

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
gamma = 0.99
explore = 20000
init_eps = 1e-1
fin_eps = 1e-4
curr_eps = init_eps
buffer_size = 50000
batch_size = 16
target_update_freq = 4
epochs = 1000
max_timesteps = 500
warm_up_steps = 250

env = build_env('Acrobot-v1')
n_state = env.observation_space.shape[0]
n_action = env.action_space.n

memory = Buffer(buffer_size)

QNet = DDQN(n_state, n_action, [64, 256]).to(device)
targetQNet = DDQN(n_state, n_action, [64, 256]).to(device)
targetQNet.load_state_dict(QNet.state_dict())
optimizer = Adam(QNet.parameters(), lr=1e-4)
lossfn = MSELoss()

print('Warming up')



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

movAvg = deque(maxlen=100)
for epoch in loop_obj:
    losses = []
    state = env.reset()[0]
    episode_rewards = []
    for steps in range(max_timesteps):
        optimizer.zero_grad()
        p = np.random.rand()
        
        if p < curr_eps:
            action = np.random.randint(0, n_action)
        else:
            state_ = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
            action = QNet.select_action(state_)

        next_state, reward, done, _, _ = env.step(action)
        memory.append((state, action, next_state, reward, done))
        episode_rewards.append(reward)
        
        if steps % target_update_freq == 0:
            targetQNet.load_state_dict(QNet.state_dict())

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
    loop_obj.set_postfix_str(f'Loss: {sum(losses)/len(losses)}, Reward: {sum(movAvg)/100}')

    if sum(movAvg)/100 >= 195:
        print(f'Converged at {epoch} epoch')
        exit()

print("Didnt Converge")