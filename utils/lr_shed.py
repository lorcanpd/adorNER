import tensorflow as tf


class LinearWarmup(tf.keras.optimizers.schedules.LearningRateSchedule):

    """
    A learning rate scheduler with the linear-warmup functionality required by
    the post-normalisation transformer architecture. Also has functionality for
    adjusting the learning rate at any time.
    ...
    Attributes
    ----------
    warmup_steps : int
        the number of training iterations in which the learning rate is
        linearly increased
    final_learning_rate : float
        the target learning rate of the linear warmup
    init_learning_rate : float
        the initial learning rate from which the learning rate is increased
    name : str
        name to be assigned to this learning rate scheduler

    Methods
    -------
    assign_lr(learning_rate)
        manually set the learning rate during training
    get_config()
        returns a dictionary with configuration information for the scheduler

    """

    def __init__(self, warmup_steps, final_learning_rate, init_learning_rate=0,
                 name=None):
        self.warmup_steps = warmup_steps
        self.final_learning_rate = final_learning_rate
        self.init_learning_rate = tf.cast(init_learning_rate, tf.float32)
        self.increment = tf.cast((final_learning_rate
                                  - init_learning_rate)
                                 / warmup_steps, tf.float32)
        self.learning_rate = tf.cast(init_learning_rate, tf.float32)
        self.name = name

    def __call__(self, step):
        with tf.name_scope(self.name or 'LinearWarmup'):
            if step < self.warmup_steps:
                self.learning_rate = (
                        self.init_learning_rate
                        + self.increment
                        * step
                )
            return self.learning_rate

    def assign_lr(self, learning_rate):
        """
        Manually changes the learning rate during training.
        ...
        Parameters
        ----------
        learning_rate : float or tf.float32
        """
        if not tf.is_tensor(learning_rate):
            self.learning_rate = tf.cast(learning_rate, tf.float32)
        else:
            self.learning_rate = learning_rate

    def get_config(self):
        """
        Retrieve the configuration parameters of the learning rate scheduler.
        ...
        Returns
        -------
        config: dict
            a dictionary with configuration information for the scheduler
        """
        config = {
            "warmup_steps": self.warmup_steps,
            "final_learning_rate": self.final_learning_rate,
            "init_learning_rate": self.init_learning_rate
        }
        return config

