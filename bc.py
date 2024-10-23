import torch
import torch.optim as optim
import numpy as np
from utils import rollout
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_bc(env, policy, expert_data, num_epochs=500, episode_length=50, 
                       batch_size=32):
    
    # Hint: Just flatten your expert dataset and use standard pytorch supervised learning code to train the policy. 
    optimizer = optim.Adam(list(policy.parameters()), lr=1e-4)
    idxs = np.array(range(len(expert_data)))
    num_batches = len(idxs)*episode_length // batch_size
    losses = []
    for epoch in range(num_epochs): 
        ## TODO Students
        np.random.shuffle(idxs)
        running_loss = 0.0
        for i in range(num_batches):
            optimizer.zero_grad()
            #========== TODO: start ==========
            # Fill in your behavior cloning implementation here

            t1_idx = np.random.randint(len(expert_data), size=(batch_size,)) # Indices of first trajectory
            t1_idx_pertraj = [np.random.randint(expert_data[c_idx]['observations'].shape[0]) for c_idx in t1_idx]
            t1_states = np.concatenate([expert_data[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
            t1_actions = np.concatenate([expert_data[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])

            t1_states = torch.Tensor(t1_states).float().to(device)
            t1_actions = torch.Tensor(t1_actions).float().to(device)

            # Compute log likelihood
            loss = - policy.log_prob(t1_states, t1_actions).mean()
            
            #========== TODO: end ==========
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        # if epoch % 10 == 0:
        print('[%d] loss: %.8f' %
            (epoch, running_loss / num_batches))
        losses.append(loss.item())

    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.title("Loss Evolution During Training", fontsize=16)
    plt.xlabel("Epoch", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()