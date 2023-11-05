import torch
import torch.nn as nn
import torch.nn.functional as F

class Extractor(nn.Module):
    def __init__(self, filters=(256, 256, 256, 128, 128), kernels=(8,)*5, input_channels=64):
        super(Extractor, self).__init__()
        self.layers = nn.ModuleList()
        for filter_, kernel in zip(filters, kernels):
            self.layers.append(nn.Conv1d(input_channels, filter_, kernel))
            self.layers.append(nn.LeakyReLU())  # LayerNorm added later in the forward method
            self.layers.append(nn.ConstantPad1d((0, kernel - 1), 0))
            input_channels = filter_
            
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Apply LayerNorm after each LeakyReLU
            if isinstance(layer, nn.LeakyReLU):
                norm_shape = x.shape[1:]
                x = nn.LayerNorm(norm_shape).to(x.device)(x)
        return x


class OutputContext(nn.Module):
    def __init__(self, filter_=64, kernel=32, input_channels=64):
        super(OutputContext, self).__init__()
        self.pad = nn.ConstantPad1d((kernel - 1, 0), 0)
        self.conv = nn.Conv1d(input_channels, filter_, kernel)
        self.activation = nn.LeakyReLU()
        # LayerNorm added later in the forward method

    def forward(self, x):
        x = self.pad(x)
        x = self.conv(x)
        norm_shape = x.shape[1:]
        x = nn.LayerNorm(norm_shape).to(x.device)(x)
        return self.activation(x)

class VLAAI(nn.Module):
    def __init__(self, nb_blocks=4, input_channels=64, output_dim=1, use_skip=True, extractor_output = 128):
        super(VLAAI, self).__init__()
        self.nb_blocks = nb_blocks
        self.use_skip = use_skip
        self.extractor = Extractor(input_channels=input_channels)
        self.dense = nn.Linear(extractor_output, input_channels)  # Equivalent of Dense in TF
        self.output_context = OutputContext(input_channels=input_channels)
        self.final_dense = nn.Linear(input_channels, output_dim)

    def forward(self, x):
        for _ in range(self.nb_blocks):
            skip = x if self.use_skip else 0
            x = self.extractor(x + skip)
            x = x.transpose(1, 2)
            x = self.dense(x)
            x = x.transpose(1, 2)
            x = self.output_context(x)
        x = x.transpose(1, 2)
        x = self.final_dense(x)
        x = x.transpose(1, 2)
        return x

def pearson_corr(y_true, y_pred):
    mean_true = y_true.mean(dim=1, keepdim=True)
    mean_pred = y_pred.mean(dim=1, keepdim=True)
    numerator = ((y_true - mean_true) * (y_pred - mean_pred)).sum(dim=1, keepdim=True)
    std_true = ((y_true - mean_true) ** 2).sum(dim=1, keepdim=True).sqrt()
    std_pred = ((y_pred - mean_pred) ** 2).sum(dim=1, keepdim=True).sqrt()
    denominator = std_true * std_pred
    return (numerator / (denominator + 1e-10))

def pearson_loss(y_true, y_pred):
    return -pearson_corr(y_true, y_pred)

def pearson_metric(y_true, y_pred):
    return pearson_corr(y_true, y_pred)

if __name__ == '__main__':
    # Example usage
    model = VLAAI()
    input_tensor = torch.rand(32, 64, 320)  # example input tensor with batch size 1, 64 channels, and length 100
    outputs = model(input_tensor)
    print("The input shape is:", input_tensor.shape)
    print("The output shape is:", outputs.shape)