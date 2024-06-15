# For the model
import torch.nn as nn
import torch 
#from utils import get_activation, get_normalization, SE_Block 
class SE_Block(nn.Module): 
    def __init__(self, channels, reduction=16, activation="relu"):
        super().__init__()
        self.reduction = reduction
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // self.reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // self.reduction, channels, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        bs, c, _, _ = x.shape
        y = self.squeeze(x).view(bs, c)
        y = self.excitation(y).view(bs, c, 1, 1)
        return x * y.expand_as(x)

#---
def get_activation(activation_name):
    if activation_name == "relu":
        return nn.ReLU6(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.ReLU6):
        return activation_name
    elif activation_name == "gelu":
        return nn.GELU()
    elif isinstance(activation_name, torch.nn.modules.activation.GELU):
        return activation_name
    elif activation_name == "leaky_relu":
        return nn.LeakyReLU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.LeakyReLU):
        return activation_name
    elif activation_name == "prelu":
        return nn.PReLU()
    elif isinstance(activation_name, torch.nn.modules.activation.PReLU):
        return activation_name
    elif activation_name == "selu":
        return nn.SELU(inplace=True)
    elif isinstance(activation_name, torch.nn.modules.activation.SELU):
        return activation_name
    elif activation_name == "sigmoid":
        return nn.Sigmoid()
    elif isinstance(activation_name, torch.nn.modules.activation.Sigmoid):
        return activation_name
    elif activation_name == "tanh":
        return nn.Tanh()
    elif isinstance(activation_name, torch.nn.modules.activation.Tanh):
        return activation_name
    elif activation_name == "mish":
        return nn.Mish()
    elif isinstance(activation_name, torch.nn.modules.activation.Mish):
        return activation_name
    else:
        raise ValueError(f"activation must be one of leaky_relu, prelu, selu, gelu, sigmoid, tanh, relu. Got: {activation_name}")

def get_normalization(normalization_name, num_channels, num_groups=32, dims=2):
    if normalization_name == "batch":
        if dims == 1:
            return nn.BatchNorm1d(num_channels)
        elif dims == 2:
            return nn.BatchNorm2d(num_channels)
        elif dims == 3:
            return nn.BatchNorm3d(num_channels)
    elif normalization_name == "instance":
        if dims == 1:
            return nn.InstanceNorm1d(num_channels)
        elif dims == 2:
            return nn.InstanceNorm2d(num_channels)
        elif dims == 3:
            return nn.InstanceNorm3d(num_channels)
    elif normalization_name == "layer":
        # return LayerNorm(num_channels)
        return nn.LayerNorm(num_channels)
    elif normalization_name == "group":
        return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    elif normalization_name == "bcn":
        if dims == 1:
            return nn.Sequential(
                nn.BatchNorm1d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 2:
            return nn.Sequential(
                nn.BatchNorm2d(num_channels),
                nn.GroupNorm(1, num_channels)
            )
        elif dims == 3:
            return nn.Sequential(
                nn.BatchNorm3d(num_channels),
                nn.GroupNorm(1, num_channels)
            )    
    elif normalization_name == "none":
        return nn.Identity()
    else:
        raise ValueError(f"normalization must be one of batch, instance, layer, group, none. Got: {normalization_name}") 

class CoreCNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, *, norm="batch", activation="relu", padding="same", residual=True):
        super(CoreCNNBlock, self).__init__()

        self.activation = get_activation(activation)
        self.residual = residual
        self.padding = padding
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.squeeze = SE_Block(self.out_channels)

        self.match_channels = nn.Identity()
        if in_channels != out_channels:
            self.match_channels = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(norm, out_channels),
            )

        self.conv1 = nn.Conv2d(self.in_channels, self.out_channels, 1, padding=0)
        self.norm1 = get_normalization(norm, self.out_channels)

        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=self.out_channels)
        self.norm2 = get_normalization(norm, self.out_channels)
        
        self.conv3 = nn.Conv2d(self.out_channels, self.out_channels, 3, padding=self.padding, groups=1)
        self.norm3 = get_normalization(norm, self.out_channels)

    def forward(self, x):
        identity = x
        x = self.activation(self.norm1(self.conv1(x)))
        x = self.activation(self.norm2(self.conv2(x)))
        x = self.norm3(self.conv3(x))
        x = x * self.squeeze(x)
        if self.residual:
            x = x + self.match_channels(identity)
        x = self.activation(x) 
        return x

class CoreEncoderBlock(nn.Module): 
    def __init__(self, depth, in_channels, out_channels, norm="batch", activation="relu", padding="same"):
        super(CoreEncoderBlock, self).__init__() 
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.blocks = []
        for i in range(self.depth): 
            _in_channels = self.in_channels if i == 0 else self.out_channels
            block = CoreCNNBlock(_in_channels, self.out_channels, norm=self.norm, activation=self.activation, padding=self.padding)

            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)
        self.downsample = nn.MaxPool2d(kernel_size=2, stride=2)
    
    def forward(self, x):
        for i in range(self.depth):
            x = self.blocks[i](x)
        before_downsample = x
        x = self.downsample(x)
        return x, before_downsample
class CoreAttentionBlock(nn.Module):
    def __init__(self,
        lower_channels,
        higher_channels, *,
        norm="batch",
        activation="relu",
        padding="same",
    ):
        super(CoreAttentionBlock, self).__init__()
        self.lower_channels = lower_channels
        self.higher_channels = higher_channels
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.expansion = 4
        self.reduction = 4
        if self.lower_channels != self.higher_channels:
            self.match = nn.Sequential(
                nn.Conv2d(self.higher_channels, self.lower_channels, kernel_size=1, padding=0, bias=False),
                get_normalization(self.norm, self.lower_channels),
            )
        self.compress = nn.Conv2d(self.lower_channels, 1, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()
        self.attn_c_pool = nn.AdaptiveAvgPool2d(self.reduction)
        self.attn_c_reduction = nn.Linear(self.lower_channels * (self.reduction ** 2), self.lower_channels * self.expansion)
        self.attn_c_extention = nn.Linear(self.lower_channels * self.expansion, self.lower_channels)

    def forward(self, x, skip):
        if x.size(1) != skip.size(1):
            x = self.match(x)
        x = x + skip
        x = self.activation(x)
        attn_spatial = self.compress(x)
        attn_spatial = self.sigmoid(attn_spatial)
        attn_channel = self.attn_c_pool(x)
        attn_channel = attn_channel.reshape(attn_channel.size(0), -1)
        attn_channel = self.attn_c_reduction(attn_channel)
        attn_channel = self.activation(attn_channel)
        attn_channel = self.attn_c_extention(attn_channel)
        attn_channel = attn_channel.reshape(x.size(0), x.size(1), 1, 1)
        attn_channel = self.sigmoid(attn_channel)
        return attn_spatial, attn_channel


class CoreDecoderBlock(nn.Module):
    def __init__(self, depth, in_channels, out_channels, *, norm="batch", activation="relu", padding="same"):
        super(CoreDecoderBlock, self).__init__()
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.activation_blocks = activation
        self.activation = get_activation(activation)
        self.norm = norm
        self.padding = padding
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.match_channels = CoreCNNBlock(self.in_channels * 2, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
        self.attention = CoreAttentionBlock(self.in_channels, self.in_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
        self.blocks = []
        for _ in range(self.depth):
            block = CoreCNNBlock(self.out_channels, self.out_channels, norm=self.norm, activation=self.activation_blocks, padding=self.padding)
            self.blocks.append(block)
        self.blocks = nn.Sequential(*self.blocks)
    
    def forward(self, x, skip):
        x = self.upsample(x)
        attn_s, attn_c = self.attention(x, skip)
        x = torch.cat([x, (skip * attn_s) + (skip + attn_c)], dim=1)
        x = self.match_channels(x)
        for i in range(self.depth):
            x = self.blocks[i](x)
        return x



class CoreEncoder(nn.Module):
    def __init__(self, *,
        input_dim=10,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ):
        super(CoreEncoder, self).__init__()
        self.depths = [3, 3, 9, 3] if depths is None else depths
        self.dims = [96, 192, 384, 768] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding
        assert len(self.depths) == len(self.dims), "depths and dims must have the same length."
        self.stem = CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding)
        self.encoder_blocks = []  
        for i in range(len(self.depths)): 
            encoder_block = CoreEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(self.dims[-1], self.output_dim),
        )

    def forward(self, x):
        x = self.stem(x)
        for block in self.encoder_blocks:
            x, _ = block(x)
        x = self.head(x)
        return x

def cutout(data):  
    min_k, max_k = 90, 170
    data = data.clone()
    h, w = data.size(2), data.size(3)
    b_size = data.size(0)
    for i in range(b_size) :
        k = (min_k + (max_k - min_k) * torch.rand(1)).long().item() 
        h_pos = ((h - k) * torch.rand(1)).long().item()
        w_pos = ((w - k) * torch.rand(1)).long().item()
        patch = data[i,:,h_pos:h_pos+k,w_pos:w_pos+k]
        patch_mean = 0.
        data[i,:,h_pos:h_pos+k,w_pos:w_pos+k] = torch.ones_like(patch) * patch_mean
    return data

class CoreUnet(nn.Module):  
    def __init__(self, *,
        input_dim=10,
        output_dim=1,
        depths=None,
        dims=None,
        activation="relu",
        norm="batch",
        padding="same",
    ): 
        super(CoreUnet, self).__init__() 
        self.depths = [3, 3, 9, 3] if depths is None else depths 
        self.dims = [96, 192, 384, 768] if dims is None else dims
        #self.depths = [3, 3, 9] if depths is None else depths
        #self.dims = [96, 192, 384] if dims is None else dims
        self.output_dim = output_dim
        self.input_dim = input_dim
        self.activation = activation
        self.norm = norm
        self.padding = padding
        self.dims = [v // 2 for v in self.dims] 
        assert len(self.depths) == len(self.dims), "depths and dims must have the same length. "   
        self.stem = nn.Sequential(
            CoreCNNBlock(self.input_dim, self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
        )  
        self.encoder_blocks = []  
        for i in range(len(self.depths)):
            encoder_block = CoreEncoderBlock(
                self.depths[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                self.dims[i],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.encoder_blocks.append(encoder_block)
        self.encoder_blocks = nn.ModuleList(self.encoder_blocks)
        self.decoder_blocks = [] 
        for i in reversed(range(len(self.encoder_blocks))):
            decoder_block = CoreDecoderBlock(
                self.depths[i],
                self.dims[i],
                self.dims[i - 1] if i > 0 else self.dims[0],
                norm=self.norm,
                activation=self.activation,
                padding=self.padding,
            )
            self.decoder_blocks.append(decoder_block)
        self.decoder_blocks = nn.ModuleList(self.decoder_blocks)
        self.bridge = nn.Sequential(
            CoreCNNBlock(self.dims[-1], self.dims[-1], norm=self.norm, activation=self.activation, padding=self.padding),
        )
        self.head = nn.Sequential(
            CoreCNNBlock(self.dims[0], self.dims[0], norm=self.norm, activation=self.activation, padding=self.padding),
            nn.Conv2d(self.dims[0], self.output_dim, kernel_size=1, padding=0),
        )

    def forward(self, x):
        skip_connections = []    
        x = self.stem(x)
        for block in self.encoder_blocks:
            x, skip = block(x)
            skip_connections.append(skip)
        x = self.bridge(x)
        return x
    

 
class MyResNet18(nn.Module):
    def __init__(self, resnet, resnet2):
        super().__init__()
        self.features = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ) 
        self.avgpool = resnet.avgpool
        self.fc = resnet.fc
        self.features2 = nn.Sequential(
            resnet2.conv1,
            resnet2.bn1,
            resnet2.relu,
            resnet2.maxpool,
            resnet2.layer1,
            resnet2.layer2,
            resnet2.layer3,
            resnet2.layer4
        )
        self.avgpool2 = resnet2.avgpool
        self.fc2 = resnet2.fc
    def _forward_impl(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x2 = self.features2(x2)
        x2 = self.avgpool2(x2)
        x2 = torch.flatten(x2, 1)
        x2 = self.fc2(x2) 
        return x, x2
    def forward(self, x: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        return self._forward_impl(x, x2)
    