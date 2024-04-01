import os
from stable_baselines3.common.callbacks import BaseCallback

class TrainAndLoggingCallback(BaseCallback):
    """
    This custom callback is used during the training of the AI model.
    It is responsible for periodically saving the model based on the frequency set by the user.
    """
    def __init__(self, check_freq, save_path, verbose=1):
        super(TrainAndLoggingCallback, self).__init__(verbose)
        self.check_freq = check_freq  # Frequency of model checkpointing.
        self.save_path = save_path    # Directory where models are saved.

    def _init_callback(self):
        # Create the save directory if it doesn't exist.
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self):
        # Save the model at the specified frequency.
        if self.n_calls % self.check_freq == 0:
            model_path = os.path.join(self.save_path, f'best_model_{self.n_calls}')
            self.model.save(model_path)
        return True
