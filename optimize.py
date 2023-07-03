from Neural_Network import Loss

class SoftmaxCrossentropyLossFunction(Loss):
    def __init__(self):

class Opmize():
    def __init__(self, lr, final_lr:float = 0, decay_type:str = 'exponential'):
        self.lr = lr
        self.final_lr = final_lr
        # step
        self.decay_type = decay_type

    def _setup_decay(self)->None:
        if self.decay_type == 'exponential':
            # step_size
            self.decay_per_epoch = ()
        else:
            self.decay_per_epoch = (self.lr - self.final_lr) / (self.max_epochs - 1)