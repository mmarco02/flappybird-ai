import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE

# Load saved rewards
eval_rewards = np.load("saved/eval_rewards.npy")

# Plot the rewards over episodes during evaluation
plt.figure(figsize=(10, 6))
plt.plot(range(len(eval_rewards)), eval_rewards, label='Evaluation Reward')
plt.xlabel("Episode")
plt.ylabel("Reward")
plt.title("Evaluation Rewards Over Episodes")
plt.legend()
plt.show()


# Load saved loss values
loss_values = np.load("saved/loss_values.npy")

# Plot the loss over episodes
plt.figure(figsize=(10, 6))
plt.plot(range(len(loss_values)), loss_values, label='Loss')
plt.xlabel("Episode")
plt.ylabel("Loss")
plt.title("Loss Over Episodes")
plt.legend()
plt.show()

# Load saved target Q-values compared to actual Q-values
target_q = np.load("saved/target_q.npy")

plt.figure(figsize=(10, 6))
for i in range(target_q.shape[1]):
    plt.plot(range(target_q.shape[0]), target_q[:, i], label=f'Action {i}')
plt.xlabel("Step")
plt.ylabel("Q-Value Estimation")
plt.title("Target Q-Values Compared to Actual Q-Values Over Training Steps")
plt.legend()
plt.show()


# Plot some sample Q-value estimations
sample_q_values = np.load("saved/sample_q_values.npy")

plt.figure(figsize=(10, 6))
for i in range(sample_q_values.shape[1]):
    plt.plot(range(sample_q_values.shape[0]), sample_q_values[:, i], label=f'Action {i}')
plt.xlabel("Step")
plt.ylabel("Q-Value Estimation")
plt.title("Sample Q-Value Estimations Over Training Steps")
plt.legend()
plt.show()

# Plot exploration probability over episodes
exploration_probs = np.load("saved/exploration_probs.npy")

plt.figure(figsize=(10, 6))
plt.plot(range(len(exploration_probs)), exploration_probs, label='Exploration Probability')
plt.xlabel("Episode")
plt.ylabel("Exploration Probability")
plt.title("Exploration Probability Decay Over Episodes")
plt.legend()
plt.show()



# Load Q-values
q_values = np.load("saved/sample_q_values.npy")

# Flatten Q-values
flat_q_values = np.concatenate(q_values, axis=0)

# Reduce dimensionality using t-SNE
tsne = TSNE(n_components=2, random_state=42)
q_values_tsne = tsne.fit_transform(flat_q_values)

# Visualize clusters
plt.scatter(q_values_tsne[:, 0], q_values_tsne[:, 1], c=range(len(q_values_tsne)), cmap='viridis', s=5)
plt.title('t-SNE Visualization of Q-Values Clusters')
plt.xlabel('t-SNE Dimension 1')
plt.ylabel('t-SNE Dimension 2')
plt.colorbar(label='Episode')
plt.show()
