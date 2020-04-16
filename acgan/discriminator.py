import torch.nn as nn


class Discriminator(nn.Module):
    def __init__(self,
                 n_classes: int,
                 n_channels: int,
                 img_size: int,
                 base_feature_maps: int = 16):
        super(Discriminator, self).__init__()

        def discriminator_block(inputs: int, outputs: int, bn: bool = True):
            """Returns layers of each discriminator block"""
            block = [
                nn.Conv2d(in_channels=inputs,
                          out_channels=outputs,
                          kernel_size=3,
                          stride=2,
                          padding=1),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Dropout2d(p=0.25)
            ]
            if bn:
                block.append(nn.BatchNorm2d(num_features=outputs, eps=0.8))
            return block

        self.conv_blocks = nn.Sequential(
            *discriminator_block(n_channels, base_feature_maps, bn=False),
            *discriminator_block(base_feature_maps, base_feature_maps*2),
            *discriminator_block(base_feature_maps*2, base_feature_maps*4),
            *discriminator_block(base_feature_maps*4, base_feature_maps*8),
        )

        # The height and width of downsampled image
        ds_size = img_size // 2 ** 4

        # Output layers
        self.adversarial_out_layer = nn.Sequential(
            nn.Linear(base_feature_maps * 8 * ds_size ** 2, 1),
            nn.Sigmoid()
        )
        self.class_out_layer = nn.Sequential(
            nn.Linear(base_feature_maps * 8 * ds_size ** 2, n_classes),
            nn.Softmax()
        )

    def forward(self, img):
        out = self.conv_blocks(img)
        out = out.view(out.shape[0], -1)
        validity = self.adversarial_out_layer(out)
        label = self.class_out_layer(out)

        return validity, label


def initialize_weights(param):
    classname = param.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(param.weight.data, mean=0.0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(param.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(param.bias.data, val=0)
