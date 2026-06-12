import numpy as np
import matplotlib.pyplot as plt


def getNextState(state, action):
    row, col = divmod(state, 4)

    if action == 0 and col > 0:
        col -= 1
    elif action == 1 and col < 3:
        col += 1
    elif action == 2 and row > 0:
        row -= 1
    elif action == 3 and row < 3:
        row += 1

    return row * 4 + col


def start():
    # Define the Environment
    n_states = 16
    n_actions = 4
    goal_state = 15

    Q_table = np.zeros((n_states, n_actions))
    print('test')

    # Set Hyperparameters
    learning_rate = 0.8
    discount_factor = 0.95
    exploration_prob = 0.2
    epochs = 1000

    # A-Learning Algorithm

    for epoch in range(epochs):
        current_state = np.random.randint(0, n_states)  

        while True:
            if np.random.rand() < exploration_prob:
                action = np.random.randint(0, n_actions)
            else:
                action = np.argmax(Q_table[current_state])

            next_state = getNextState(current_state, action)

            reward = 1 if next_state == goal_state else 0

            Q_table[current_state, action] += learning_rate * (
                reward + discount_factor * np.max(Q_table[next_state]) -
                Q_table[current_state, action]
            )

            if next_state == goal_state:
                break

            current_state = next_state

    # Output
    q_values_grid = np.max(Q_table, axis=1).reshape((4, 4))

    plt.figure(figsize=(6, 6))
    plt.imshow(q_values_grid, cmap='coolwarm', interpolation='nearest')
    plt.colorbar(label='Q-value')
    plt.title('Learned Q-values for Each State')
    plt.xticks(np.arange(4), ['0', '1', '2', '3'])
    plt.yticks(np.arange(4), ['0', '1', '2', '3'])
    plt.gca().invert_yaxis()
    plt.grid(True)

    for i in range(4):
        for j in range(4):
            plt.text(j, i, f'{q_values_grid[i, j]:.2f}', ha='center',
                     va='center', color='black')

    plt.show()

    print("Learned Q-table:")
    print(Q_table)


def get_moving_avgs(arr, window, convolution_mode):
    return np.convolve(
        np.array(arr).flatten(),
        np.ones(window),
        mode=convolution_mode
    ) / window
    
    # Smooth over a 500-episode window
    rolling_length = 500
    fig, axs = plt.subplots(ncols=3, figsize=(12, 5))
    
    print('test')
    # Episode rewards (win/loss performance)
    axs[0].set_title("Episode rewards")
    reward_moving_average = get_moving_avgs(
        env.return_queue,
        rolling_length,
        "valid"
    )
    axs[0].plot(range(len(reward_moving_average)), reward_moving_average)
    axs[0].set_ylabel("Average Reward")
    axs[0].set_xlabel("Episode")
    
    # Episode lengths (how many actions per hand)
    axs[1].set_title("Episode lengths")
    length_moving_average = get_moving_avgs(
        env.length_queue,
        rolling_length,
        "valid"
    )
    axs[1].plot(range(len(length_moving_average)), length_moving_average)
    axs[1].set_ylabel("Average Episode Length")
    axs[1].set_xlabel("Episode")
    
    # Training error (how much we're still learning)
    axs[2].set_title("Training Error")
    training_error_moving_average = get_moving_avgs(
        agent.training_error,
        rolling_length,
        "same"
    )
    axs[2].plot(range(len(training_error_moving_average)), training_error_moving_average)
    axs[2].set_ylabel("Temporal Difference Error")
    axs[2].set_xlabel("Step")
    
    plt.tight_layout()
    plt.show()
    