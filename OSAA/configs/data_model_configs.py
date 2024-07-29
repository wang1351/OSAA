def get_dataset_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]

class PU():
    def __init__(self):
        super(PU, self)
        self.scenarios = [("B","D"),("A", "E"), ("C", "F")]
        self.class_names = ['healthy','IF','OF']
        self.sequence_len = 5120
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        # model configs
        self.input_channels = 1
        self.kernel_size = 9
        self.stride = 1
        self.dropout = 0.4
        self.num_classes = 3

        self.mid_channels = 64 
        self.final_out_channels = 128
        self.features_len = 1
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
        self.decodersize = 640

class CWRU():
    def __init__(self):
        super(CWRU, self)
        self.scenarios =[("FE1772","DE1797"),("DE1772", "FE1797"), ("FE1730", "DE1750")]
        self.class_names = ['healthy','IF','OF','B']
        self.sequence_len = 1024
        self.shuffle = True
        self.drop_last = True
        self.normalize = True

        self.input_channels = 1
        self.kernel_size = 9
        self.stride = 1
        self.dropout = 0.4
        self.num_classes = 4

        self.mid_channels = 64
        self.final_out_channels = 128
        self.features_len = 1
        self.disc_hid_dim = 64
        self.hidden_dim = 500
        self.DSKN_disc_hid = 128
        self.decodersize = 128

