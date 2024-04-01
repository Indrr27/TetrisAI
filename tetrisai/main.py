from callbacks import TrainAndLoggingCallback
from environment_setup import create_tetris_env
from stable_baselines3 import PPO

# Directories for storing checkpoints and logs during training.
CHECKPOINT_DIR = './train/'
LOG_DIR = './logs/'

# Initialize the custom Tetris environment for the AI to interact with.
env = create_tetris_env()

# Set up a callback for periodic saving of the model during training.
callback = TrainAndLoggingCallback(check_freq=10000, save_path=CHECKPOINT_DIR)

# Configure and initiate the Proximal Policy Optimization (PPO) model.
# This involves specifying the policy type, the environment, various training parameters,
# and where to log training statistics.
model = PPO('CnnPolicy', env, verbose=1, tensorboard_log=LOG_DIR, learning_rate=0.00005, n_steps=512)
model.learn(total_timesteps=4000000, callback=callback)

# Save the trained model for later use or evaluation.
model.save('thisisatestmodel')

# Load the best-performing model based on training results.
#model = PPO.load('./train/best_model_10000')

# Prepare the environment and model for the game simulation.
state = env.reset()
done = True

# Execute a loop where the model continuously plays Tetris, making decisions based on its training.
# This section can be modified to control the number of game steps to simulate.
for step in range(100000):
    if done:
        state = env.reset()  # Reset the game state if the previous game is over.
    action, _ = model.predict(state)
    state, reward, done, info = env.step(action)  # Execute the chosen action in the environment.
    env.render()  # Display the game screen. This can be disabled during headless training sessions.

env.close()  # Properly close the environment to free up resources.
