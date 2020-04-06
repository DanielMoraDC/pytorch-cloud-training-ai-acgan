import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_channels: int,
                 img_size: int,
                 latent_dimensions: int,
                 base_feature_maps: int = 64):
        super(Generator, self).__init__()
        self._base_feature_maps = base_feature_maps
        self.label_embedding = nn.Embedding(n_classes, latent_dimensions)
        self.init_size = img_size // 4  # Initial size before upsampling
        self.input_layer = nn.Sequential(
            nn.Linear(latent_dimensions, base_feature_maps * self.init_size ** 2)
        )

        def generator_block(inputs: int, outputs: int):
            return [
                nn.Conv2d(in_channels=inputs,
                          out_channels=outputs,
                          kernel_size=3,
                          stride=1,
                          padding=1),
                nn.BatchNorm2d(num_features=outputs, eps=0.8),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)
            ]

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(num_features=base_feature_maps),
            nn.Upsample(scale_factor=2),
            *generator_block(base_feature_maps, base_feature_maps*2),
            nn.Upsample(scale_factor=2),
            *generator_block(base_feature_maps * 2, base_feature_maps),
            nn.Conv2d(in_channels=base_feature_maps,
                      out_channels=n_channels,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.Tanh(),
        )

    def forward(self, noise, labels):
        input = torch.mul(self.label_embedding(labels.long()), noise)
        input_tensor = self.input_layer(input)
        resized_input_tensor = input_tensor.view(input_tensor.shape[0],
                                                 self._base_feature_maps,
                                                 self.init_size,
                                                 self.init_size)
        return self.conv_blocks(resized_input_tensor)


def initialize_weights(param):
    classname = param.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(param.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(param.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(param.bias.data, val=0)
