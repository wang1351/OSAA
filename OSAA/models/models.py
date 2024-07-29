import torch
from torch import nn
import math
from torch.autograd import Function
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import pdb

# from utils import weights_init

def get_backbone_class(backbone_name):
    """Return the algorithm class with the given name."""
    if backbone_name not in globals():
        raise NotImplementedError("Algorithm not found: {}".format(backbone_name))
    return globals()[backbone_name]


##################################################
##########  BACKBONE NETWORKS  ###################
##################################################
class CNN(nn.Module):
    def __init__(self, configs):
        super(CNN, self).__init__()

        self.conv_block1 = nn.Sequential(
            nn.Conv1d(configs.input_channels, configs.mid_channels, kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        self.pool1 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=True)

        self.conv_block2 = nn.Sequential(
            nn.Conv1d(configs.mid_channels, configs.mid_channels * 2, kernel_size=8, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU()
        )
        self.pool2 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=True)

        self.conv_block3 = nn.Sequential(
            nn.Conv1d(configs.mid_channels * 2, configs.final_out_channels, kernel_size=8, stride=1, bias=False,
                      padding=4),
            nn.BatchNorm1d(configs.final_out_channels),
            nn.ReLU()
        )
        self.pool3 = nn.MaxPool1d(kernel_size=2, stride=2, padding=0,return_indices=True)

        self.adaptive_pool = nn.AdaptiveAvgPool1d(configs.features_len)

    def forward(self, x_in):

        x = self.conv_block1(x_in)
        x, ind1 = self.pool1(x)
        x = self.conv_block2(x)
        x, ind2 = self.pool1(x)
        x = self.conv_block3(x)
        x, ind3 = self.pool1(x)
        x = self.adaptive_pool(x)
        x_flat = x.reshape(x.shape[0], -1)
        return x_flat, [ind1, ind2,ind3]
    
class CNNDecoder(nn.Module):
    def __init__(self, configs):
        super(CNNDecoder, self).__init__()

        self.revconv_block3 = nn.Sequential(
            nn.ConvTranspose1d(configs.mid_channels,configs.input_channels,  kernel_size=configs.kernel_size,
                      stride=configs.stride, bias=False, padding=(configs.kernel_size // 2)),
            nn.BatchNorm1d(configs.input_channels),
            nn.ReLU()
        )
        self.pool3 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        
        
        self.revconv_block2 = nn.Sequential(
            nn.ConvTranspose1d(configs.mid_channels * 2, configs.mid_channels,  kernel_size=9, stride=1, bias=False, padding=4),
            nn.BatchNorm1d(configs.mid_channels),
            nn.ReLU()
        )
        self.pool2 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)
        
        
        self.revconv_block1 = nn.Sequential(
            nn.ConvTranspose1d(configs.final_out_channels,configs.mid_channels * 2,  kernel_size=9, stride=1, bias=False, 
                      padding=4),
            nn.BatchNorm1d(configs.mid_channels * 2),
            nn.ReLU(),
            nn.Dropout(configs.dropout)
        )
        
        self.pool1 = nn.MaxUnpool1d(kernel_size=2, stride=2, padding=0)


        self.expand = nn.Linear(1, configs.decodersize) #5120: 640 1024:128

    def forward(self, x_in, indlist):
        ind1, ind2, ind3 = indlist
      #  pdb.set_trace()
        x = self.expand(x_in)
        x = self.pool1(x, ind3)
        x = self.revconv_block1(x)
        x = self.pool2(x, ind2)
        x = self.revconv_block2(x)
        x = self.pool3(x, ind1)

        x = self.revconv_block3(x)
        

        return x


class classifier(nn.Module):
    def __init__(self, configs):
        super(classifier, self).__init__()

        model_output_dim = configs.features_len
        self.logits = nn.Linear(model_output_dim * configs.final_out_channels, configs.num_classes)

    def forward(self, x):
        predictions = self.logits(x)
        return predictions

##################################################
##########  OTHER NETWORKS  ######################
##################################################
class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


class RandomLayer(nn.Module):
    def __init__(self, input_dim_list=[], output_dim=1024):
        super(RandomLayer, self).__init__()
        self.input_num = len(input_dim_list)
        self.output_dim = output_dim
        self.random_matrix = [torch.randn(input_dim_list[i], output_dim) for i in range(self.input_num)]

    def forward(self, input_list):
        return_list = [torch.mm(input_list[i], self.random_matrix[i]) for i in range(self.input_num)]
        return_tensor = return_list[0] / math.pow(float(self.output_dim), 1.0 / len(return_list))
        for single in return_list[1:]:
            return_tensor = torch.mul(return_tensor, single)
        return return_tensor

    def cuda(self):
        super(RandomLayer, self).cuda()
        self.random_matrix = [val.cuda() for val in self.random_matrix]


class Discriminator_(nn.Module):

    def __init__(self, configs):
        """Init discriminator."""
        super(Discriminator_, self).__init__()

        self.restored = False

        self.layer = nn.Sequential(
            nn.Linear(configs.features_len * configs.final_out_channels * configs.num_classes, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, configs.disc_hid_dim),
            nn.ReLU(),
            nn.Linear(configs.disc_hid_dim, 2)
            # nn.LogSoftmax(dim=1)
        )

    def forward(self, input):
        """Forward the discriminator."""
        out = self.layer(input)
        return out

