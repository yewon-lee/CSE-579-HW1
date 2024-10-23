import torch
import torch.optim as optim
import numpy as np

from utils import rollout, relabel_action

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def simulate_policy_dagger(env, policy, expert_paths, expert_policy=None, num_epochs=500, episode_length=50,
                            batch_size=32, num_dagger_iters=10, num_trajs_per_dagger=10):
    
    
    # Fill in your dagger implementation here. 
    # Hint: Loop through num_dagger_iters iterations, at each iteration train a policy on the current dataset.
    # Then rollout the policy, use relabel_action to relabel the actions along the trajectory with "expert_policy" and then add this to current dataset
    # Repeat this so the dataset grows with states drawn from the policy, and relabeled actions using the expert.
    
    
    # Optimizer code
    optimizer = optim.Adam(list(policy.parameters()))
    losses = []
    returns = []

    trajs = expert_paths
    # Dagger iterations
    for dagger_itr in range(num_dagger_iters):
        idxs = np.array(range(len(trajs)))
        num_batches = len(idxs)*episode_length // batch_size
        losses_inner = []
        # Train the model with Adam
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(num_batches):
                optimizer.zero_grad()
                #========== TODO: begin ==========
                # Fill in your behavior cloning implementation here

                t1_idx = np.random.randint(len(trajs), size=(batch_size,)) # Indices of first trajectory
                t1_idx_pertraj = [np.random.randint(trajs[c_idx]['observations'].shape[0]) for c_idx in t1_idx]
                t1_states = np.concatenate([trajs[c_idx]['observations'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
                t1_actions = np.concatenate([trajs[c_idx]['actions'][t_idx][None] for (c_idx, t_idx) in zip(t1_idx, t1_idx_pertraj)])
                t1_states = torch.Tensor(t1_states).float().to(device)
                t1_actions = torch.Tensor(t1_actions).float().to(device)

                # Compute log likelihood
                loss = - policy.log_prob(t1_states, t1_actions).mean()

                #========== TODO: end ==========
                loss.backward()
                optimizer.step()

                # print statistics
                running_loss += loss.item()
            # if epoch % 10 == 0:
            print('[%d, %5d] loss: %.8f' %(epoch + 1, i + 1, running_loss/num_batches))
            losses_inner.append(running_loss/num_batches)
        # losses.append(sum(losses_inner)/len(losses_inner))
        losses.append(losses_inner[-1])

        # Collecting more data for dagger
        trajs_recent = []
        for k in range(num_trajs_per_dagger):
            env.reset()
            #========== TODO: start ==========
            # Rollout the policy on the environment to collect more data, relabel them, add them into trajs_recent
          
            rollouts = rollout(env, policy, 'dagger', episode_length, render=False)
            path = relabel_action(rollouts, expert_policy)
            trajs_recent.append(path)
            
            #========== TODO: end ==========

        trajs += trajs_recent
        mean_return = np.mean(np.array([traj['rewards'].sum() for traj in trajs_recent]))
        print("Average DAgger return is " + str(mean_return))
        returns.append(mean_return)
    
    import matplotlib.pyplot as plt
    plt.figure(figsize=(8,6))
    plt.plot(losses, label="Training Loss", color="blue")
    plt.title("Average Loss at each DAgger Iteration", fontsize=16)
    plt.xlabel("DAgger Iteration", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True)
    plt.legend()
    plt.show()
