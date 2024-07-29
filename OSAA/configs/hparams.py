def get_hparams_class(dataset_name):
    """Return the algorithm class with the given name."""
    if dataset_name not in globals():
        raise NotImplementedError("Dataset not found: {}".format(dataset_name))
    return globals()[dataset_name]
        
class PU():
    def __init__(self):
        super(PU, self).__init__()
        self.train_params = {
                'num_epochs': 1,
                'batch_size': 64,
                'weight_decay': 1e-5
        }
        self.alg_hparams = {
            'DISTANT':{'learning_rate': 1e-4,   'clasweight':1,  'domainweight': 1,    'threshold_s':0.5,  'threshold_i': 0.5}}

class CWRU():
    def __init__(self):
        super(CWRU, self).__init__()
        self.train_params = {
                'num_epochs': 5,
                'batch_size': 64,
                'weight_decay': 1e-5
        }
        self.alg_hparams = {

            'DISTANT':{'learning_rate': 1e-4,   'clasweight':1,  'domainweight': 0.3,    'threshold_s':0.5,  'threshold_i': 0.5}}
        


