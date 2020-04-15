class Struct:

    def __init__(self, **entries):

        self.__dict__.update(entries)

class Config:

    def __init__(self):

        self.verbose = True

        # load image to this size
        self.load_size = 286

        # crop image to this size
        self.crop_size = 256

        self.batch_size = 1

        self.learning_rate = 0.0001

        self.beta_1 = 0.5

        self.adversarial_loss = 'lsgan '# ['gan', 'hinge_v1', 'hinge_v2', 'lsgan', 'wgan']

        self.gradient_penalty = None # [None, 'dragan', 'wgan-gp']

        self.gradient_penalty_weight = 10.0

        self.cycle_loss_weight = 10.0

        self.identity_loss_weight = 0.0

        self.pool_size = 50

        self.model_name = None

        self.dataset = 'monet2photo'

        self.seed = 2020

        self.num_prefetch_batch = 1

        self.titles = ['Original', 'Translated', 'Reconstructed']

        self.r = 2
        
        self.c = 3