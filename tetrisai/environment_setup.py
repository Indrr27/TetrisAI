import gym_tetris
from nes_py.wrappers import JoypadSpace
from gym_tetris.actions import MOVEMENT
from gym.wrappers import GrayScaleObservation
from stable_baselines3.common.vec_env import VecFrameStack, DummyVecEnv

def create_tetris_env():
    """
    This function sets up the Tetris environment for the AI. It configures the environment
    for grayscale and stacks frames to provide a temporal dimension to the observations.
    """
    # Modify the JoypadSpace wrapper to ensure compatibility with environment resets.
    JoypadSpace.reset = lambda self, **kwargs: self.env.reset(**kwargs)

    # Initialize the Tetris environment with specific settings for API compatibility and rendering.
    env = gym_tetris.make('TetrisA-v2', apply_api_compatibility=True, render_mode='human')

    # Apply a predefined action space to the environment to restrict the AI's possible actions.
    env = JoypadSpace(env, MOVEMENT)

    # Convert environment images to grayscale while keeping the dimensionality.
    env = GrayScaleObservation(env, keep_dim=True)

    # Wrap the environment to work with vectorized actions and stack frames.
    env = DummyVecEnv([lambda: env])  # Convert to a vectorized environment.
    env = VecFrameStack(env, 4, channels_order='last')  # Stack 4 frames to give the model a sense of motion.

    return env
