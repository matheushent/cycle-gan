"""Core module for callbacks related ops"""
import tensorflow as tf

from src import join

class Callbacks:

    def __init__(self, output_dir, model):
        
        self.output_dir = output_dir
        self.model = model
        self.tensorboard = self.TensorBoardCallback()
        self.early_stopping = self.EarlyStoppingCallback()
        self.model_checkpoint = self.ModelCheckpointCallback()

    def TensorBoardCallback(self):
        """Utility function to instantiate TensorBoard Callback

        Returns:
            tf.keras.callbacks.Tensorboard instantiated and set up.
        """

        callback = tf.keras.callbacks.TensorBoard(
            self.output_dir
        )

        callback.set_model(self.model)

        return callback

    def EarlyStoppingCallback(self):
        """Utility function to instantiate EarlyStopping Callback

        Returns:
            tf.keras.callbacks.EarlyStopping instantiated and set up.
        """

        callback = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            verbose=1,
            restore_best_weights=True
        )

        callback.set_model(self.model)

        return callback

    def ModelCheckpointCallback(self):
        """Utility function to instantiate ModelCheckpoint Callback

        Returns:
            tf.keras.callbacks.ModelCheckpoint instantiated and set up.
        """

        callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=join(self.output_dir, 'model.h5'),
            verbose=1,
            save_best_only=True,
            save_weights_only=True
        )

        callback.set_model(self.model)

        return callback

class LinearDecay(tf.keras.optimizers.schedules.LearningRateSchedule):
    """Utility class to define the learning rate through time

    Args:
        init_learning_rate (float): The initial learning rate
        epochs (int): Epochs to run training
        epoch_decay (int): Epoch to start decaying learning rate
        beta_1 (float): The exponential decay rate for the 1st moment estimates
    """

    def __init__(self, init_learning_rate, epochs, epoch_decay, beta_1):
        super(LinearDecay, self).__init__()
        self.init_learning_rate = init_learning_rate
        self.epochs = epochs
        self.epoch_decay = epoch_decay
        self.beta_1 = beta_1
        self.current_learning_rate = tf.Variable(
            initial_value=init_learning_rate,
            trainable=False, dtype=tf.float32
        )

        # define functions to call
        self.true_fn = lambda step: self.init_learning_rate * (1 - (1 / ((self.epochs - self.epoch_decay) * (step - self.epoch_decay))))
        self.false_fn = lambda: self.init_learning_rate

    def __call__(self, step):
        self.current_learning_rate.assign(tf.cond(
            step >= self.epoch_decay,
            true_fn=self.true_fn(step),
            false_fn=self.false_fn
        ))