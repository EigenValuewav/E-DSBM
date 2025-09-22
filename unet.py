import torch
from torch import nn
import math
import torch.nn.functional as F

def exists(x):
    return x is not None

def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8, kernel_size=3):
        super().__init__()
        if isinstance(kernel_size, int):
            padding = kernel_size // 2
        else:
            padding = tuple(k // 2 for k in kernel_size)
        self.proj = nn.Conv2d(dim, dim_out, kernel_size, padding=padding)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()
    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)
        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift
        x = self.act(x)
        return x

class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, cond_emb_dim=None, groups=8, kernel_size=3):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.SiLU(), nn.Linear(cond_emb_dim, dim_out * 2))
            if exists(cond_emb_dim)
            else None
        )
        self.block1 = Block(dim, dim_out, groups=groups, kernel_size=kernel_size)
        self.block2 = Block(dim_out, dim_out, groups=groups, kernel_size=kernel_size)
        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, cond_emb=None):
        scale_shift = None
        if exists(self.mlp) and exists(cond_emb):
            cond_emb = self.mlp(cond_emb)
            cond_emb = cond_emb.view(cond_emb.shape[0], -1, 1, 1)
            scale_shift = cond_emb.chunk(2, dim=1)
        
        identity = self.res_conv(x)
        h = self.block1(x, scale_shift=scale_shift)
        h = self.block2(h)
        return h + identity

class FrequencySplitProcessor(nn.Module):
    def __init__(self, dim, split_bin, cond_emb_dim, groups=8):
        super().__init__()
        self.split_bin = split_bin
        self.lf_processor = ResnetBlock(dim, dim, cond_emb_dim=cond_emb_dim, groups=groups, kernel_size=(1, 3))
        self.hf_processor = ResnetBlock(dim, dim, cond_emb_dim=cond_emb_dim, groups=groups, kernel_size=(1, 3))

    def forward(self, x, cond_emb):
        lf_feat = x[:, :, :self.split_bin, :]
        hf_feat = x[:, :, self.split_bin:, :]
        lf_feat_processed = self.lf_processor(lf_feat, cond_emb)
        hf_feat_processed = self.hf_processor(hf_feat, cond_emb)
        return torch.cat((lf_feat_processed, hf_feat_processed), dim=2)

def Upsample(dim_in, dim_out):
    return nn.ConvTranspose2d(dim_in, dim_out, 4, 2, 1)

def Downsample(dim_in, dim_out):
    return nn.Conv2d(dim_in, dim_out, 4, 2, 1)


class ConditionalUnet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4),
        channels=2,
        self_condition=True,
        resnet_block_groups=8,
        split_freq_bin=12,
        envelope_emb_dim=32
    ):
        super().__init__()
        self.self_condition = self_condition
        self.split_freq_bin = split_freq_bin

        input_channels = channels * (2 if self_condition else 1)
        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(dim),
            nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )
        
        self.envelope_mlp = nn.Sequential(
            nn.Linear(1, envelope_emb_dim), nn.GELU(), nn.Linear(envelope_emb_dim, envelope_emb_dim)
        )
        cond_emb_dim = time_dim + envelope_emb_dim

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        # Encoder (Downsampling)
        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                nn.ModuleList([
                    ResnetBlock(dim_in, dim_out, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups),
                    ResnetBlock(dim_out, dim_out, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups),
                    Downsample(dim_out, dim_out) if not is_last else nn.Identity(),
                ])
            )

        # Bottleneck
        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups)
        self.mid_processor = FrequencySplitProcessor(mid_dim, split_freq_bin, cond_emb_dim, resnet_block_groups)
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups)

        # Decoder(Upsampling)
        in_ch = mid_dim
        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            self.ups.append(
                nn.ModuleList([
                    ResnetBlock(dim_out * 2, dim_out, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups),
                    ResnetBlock(dim_out, dim_out, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups),
                    Upsample(in_ch, dim_out),
                ])
            )
            in_ch = dim_out

        # final Layer
        self.out_dim = default(out_dim, channels)
        self.final_res_block = ResnetBlock(init_dim * 2, dim, cond_emb_dim=cond_emb_dim, groups=resnet_block_groups)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

        self.envelope_head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, None)),
            nn.Flatten(start_dim=2),
            nn.Conv1d(dim, 1, kernel_size=1),
            nn.Flatten(start_dim=1)
        )

    def forward(self, x, time, x_cond, envelope=None):
        if self.self_condition:
            x = torch.cat((x_cond, x), dim=1)

        x = self.init_conv(x)
        r = x.clone()

        t_emb = self.time_mlp(time)
        
        if exists(envelope):
            env_emb_temporal = self.envelope_mlp(envelope.unsqueeze(-1))
            env_emb_global = env_emb_temporal.mean(dim=1)
            
            if env_emb_global.dim() == 1:
                env_emb_global = env_emb_global.unsqueeze(0)
            cond_emb = torch.cat([t_emb, env_emb_global], dim=1)
        else:
            padding = torch.zeros(t_emb.shape[0], self.envelope_mlp[-1].out_features, device=x.device)
            cond_emb = torch.cat([t_emb, padding], dim=1)

        h = []
        for block1, block2, downsample in self.downs:
            x = block1(x, cond_emb)
            x = block2(x, cond_emb)
            h.append(x)
            x = downsample(x)

        x = self.mid_block1(x, cond_emb)
        x = self.mid_processor(x, cond_emb)
        x = self.mid_block2(x, cond_emb)
        
        # Decoder Forward Pass
        for block1, block2, upsample in self.ups:
            # Skip connection concat, Resnetblock pass
            x = upsample(x)
            skip_connection = h.pop()
            
            target_size = x.shape[2:]
            skip_connection = F.interpolate(skip_connection, size=target_size, mode='bilinear', align_corners=False)
            
            x = torch.cat((x, skip_connection), dim=1)
            x = block1(x, cond_emb)
            x = block2(x, cond_emb)

        # interpolate
        target_size = r.shape[2:]
        x = F.interpolate(x, size=target_size, mode='bilinear', align_corners=False)
        
        x = torch.cat((x, r), dim=1)
        x = self.final_res_block(x, cond_emb)

        predicted_drift = self.final_conv(x)
        predicted_envelope = self.envelope_head(x)

        return predicted_drift, predicted_envelope