## PromptIR: Prompting for All-in-One Blind Image Restoration
## Vaishnav Potlapalli, Syed Waqas Zamir, Salman Khan, and Fahad Shahbaz Khan
## https://arxiv.org/abs/2306.13090

import torch
import torch.nn as nn
import torch.nn.functional as F
import numbers

from einops import rearrange


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)
        assert len(normalized_shape) == 1
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm, self).__init__()
        if LayerNorm_type == 'BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class FeedForward(nn.Module):
    """Gated-Dconv Feed-Forward Network (GDFN)"""
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()
        hidden_features = int(dim * ffn_expansion_factor)
        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)
        self.dwconv = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3,
                                stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.project_out = nn.Conv2d(hidden_features, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1, x2 = self.dwconv(x).chunk(2, dim=1)
        x = F.gelu(x1) * x2
        return self.project_out(x)


class Attention(nn.Module):
    """Multi-DConv Head Transposed Self-Attention (MDTA)"""
    def __init__(self, dim, num_heads, bias):
        super(Attention, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.qkv = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=bias)
        self.qkv_dwconv = nn.Conv2d(dim * 3, dim * 3, kernel_size=3,
                                    stride=1, padding=1, groups=dim * 3, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        b, _, h, w = x.shape
        qkv = self.qkv_dwconv(self.qkv(x))
        q, k, v = qkv.chunk(3, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        attn = attn.softmax(dim=-1)
        out = rearrange(attn @ v, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class Downsample(nn.Module):
    def __init__(self, n_feat):
        super(Downsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelUnshuffle(2))

    def forward(self, x):
        return self.body(x)


class Upsample(nn.Module):
    def __init__(self, n_feat):
        super(Upsample, self).__init__()
        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat * 2, kernel_size=3, stride=1, padding=1, bias=False),
            nn.PixelShuffle(2))

    def forward(self, x):
        return self.body(x)


class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, ffn_expansion_factor, bias, LayerNorm_type):
        super(TransformerBlock, self).__init__()
        self.norm1 = LayerNorm(dim, LayerNorm_type)
        self.attn = Attention(dim, num_heads, bias)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class OverlapPatchEmbed(nn.Module):
    """Overlapped image patch embedding with 3x3 Conv"""
    def __init__(self, in_c=3, embed_dim=48, bias=False):
        super(OverlapPatchEmbed, self).__init__()
        self.proj = nn.Conv2d(in_c, embed_dim, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        return self.proj(x)


class Topm_CrossAttention_Restormer(nn.Module):
    def __init__(self, dim, num_heads, bias):
        super(Topm_CrossAttention_Restormer, self).__init__()
        self.num_heads = num_heads
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))
        self.kv = nn.Conv2d(dim, dim * 2, kernel_size=1, bias=bias)
        self.kv_dwconv = nn.Conv2d(dim * 2, dim * 2, kernel_size=3,
                                   stride=1, padding=1, groups=dim * 2, bias=bias)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.q_dwconv = nn.Conv2d(dim, dim, kernel_size=3,
                                  stride=1, padding=1, groups=dim, bias=bias)
        self.project_out = nn.Conv2d(dim, dim, kernel_size=1, bias=bias)
        self.attn4 = nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, x_q, x_kv):
        b, _, h, w = x_q.shape
        q = self.q_dwconv(self.q(x_q))
        kv = self.kv_dwconv(self.kv(x_kv))
        k, v = kv.chunk(2, dim=1)
        q = rearrange(q, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k = rearrange(k, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v = rearrange(v, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        q = torch.nn.functional.normalize(q, dim=-1)
        k = torch.nn.functional.normalize(k, dim=-1)
        _, _, C, _ = q.shape
        mask4 = torch.zeros(b, self.num_heads, C, C, device=x_q.device, requires_grad=False)
        attn = (q @ k.transpose(-2, -1)) * self.temperature
        index = torch.topk(attn, k=int(C * 9 / 10), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))
        attn4 = attn4.softmax(dim=-1)
        out = rearrange(attn4 @ v * self.attn4,
                        'b head c (h w) -> b (head c) h w', head=self.num_heads, h=h, w=w)
        return self.project_out(out)


class ch_shuffle_high_text(nn.Module):
    def __init__(self, ch_dim, num_heads, LayerNorm_type, ffn_expansion_factor, bias, lin_ch=512):
        super(ch_shuffle_high_text, self).__init__()
        self.dim = ch_dim
        self.linear_layer1 = nn.Linear(lin_ch, lin_ch)
        self.linear_layer3 = nn.Linear(lin_ch, 2 * ch_dim)
        self.conv1x1 = nn.Conv2d(ch_dim, 2 * ch_dim, kernel_size=1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(2 * ch_dim, ch_dim, kernel_size=1, stride=1, padding=0)
        self.norm1 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm2 = LayerNorm(ch_dim, LayerNorm_type)
        self.norm3 = LayerNorm(ch_dim, LayerNorm_type)
        self.select_attn = Topm_CrossAttention_Restormer(ch_dim, num_heads, bias=False)
        self.ffn = FeedForward(ch_dim, ffn_expansion_factor, bias)

    def forward(self, img_featur, text_code):
        b, _, _, _ = img_featur.shape
        img_feature2 = img_featur
        text_code = self.linear_layer1(text_code)
        text_code = self.linear_layer3(text_code)
        _, soft_indices = torch.topk(text_code, k=2 * self.dim)
        img_featur = self.conv1x1(img_featur)
        shuffled_img = img_featur[torch.arange(b).unsqueeze(1), soft_indices, :, :]
        q = self.conv_out(shuffled_img)
        att = self.select_attn(self.norm1(q), self.norm2(img_feature2))
        output = att + self.ffn(self.norm3(att))
        return output, img_feature2


class DegradationEstimator(nn.Module):
    """Estimates spatial severity map and degradation type probabilities from the degraded image."""
    def __init__(self, num_types=7):
        super().__init__()
        self.shared_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
        )
        self.severity_head = nn.Sequential(
            nn.Conv2d(32, 1, 1), nn.Sigmoid(),
        )
        # Wider/deeper branch dedicated to type classification: haze and blur both
        # wash out high-frequency detail, so a larger receptive field plus
        # contrast (std) statistics are needed on top of the plain channel mean
        # to tell them apart from the shared 32-ch encoder features.
        self.type_encoder = nn.Sequential(
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
        )
        self.type_head = nn.Sequential(
            nn.Linear(64 * 2, 64), nn.ReLU(),
            nn.Linear(64, num_types),
        )

    def forward(self, x):
        features = self.shared_encoder(x)                     # (B, 32, H, W)
        severity = self.severity_head(features)               # (B, 1, H, W)
        type_features = self.type_encoder(features.detach())  # (B, 64, H/2, W/2)
        pooled_mean = type_features.mean(dim=[2, 3])                         # (B, 64)
        pooled_var  = type_features.var(dim=[2, 3], unbiased=False)
        pooled_std  = torch.sqrt(pooled_var + 1e-6)                          # (B, 64)
        pooled      = torch.cat([pooled_mean, pooled_std], dim=1)            # (B, 128)
        type_logits = self.type_head(pooled)                                 # (B, num_types)
        return severity, type_logits

class SeveritySpatialAttention(nn.Module):
    """Weights skip-connection features spatially using the severity map from DegradationEstimator."""
    def __init__(self, feature_dim):
        super().__init__()
        self.conv = nn.Conv2d(feature_dim + 1, feature_dim, kernel_size=1)

    def forward(self, features, severity_map):
        sev = F.interpolate(severity_map, size=features.shape[-2:], mode='bilinear', align_corners=False)
        weights = torch.sigmoid(self.conv(torch.cat([features, sev], dim=1)))
        return features * weights
    
class FiLM(nn.Module):
    """Feature-wise Linear Modulation — scales and shifts feature maps using severity scalar."""
    def __init__(self, feature_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(1, 64), nn.ReLU(),
            nn.Linear(64, 2 * feature_dim),
        )

    def forward(self, features, severity_scalar):
        params = self.mlp(severity_scalar)                   # (B, 2*C)
        gamma, beta = params.chunk(2, dim=-1)
        gamma = gamma.unsqueeze(-1).unsqueeze(-1)            # (B, C, 1, 1)
        beta  = beta.unsqueeze(-1).unsqueeze(-1)
        return gamma * features + beta




class ChannelShuffle_skip_textguaid(nn.Module):
    def __init__(self,
                 inp_channels=3,
                 out_channels=3,
                 dim=48,
                 num_blocks=[4, 6, 6, 8],
                 num_refinement_blocks=4,
                 heads=[1, 2, 4, 8],
                 ffn_expansion_factor=2.66,
                 bias=False,
                 LayerNorm_type='WithBias',
                 text_features_all=None,
                 num_types=7,
                 use_deg_estimator=True,
                 use_spatial_attn=True,
                 use_film=True,
                 use_text_code=True):
        super(ChannelShuffle_skip_textguaid, self).__init__()
        self.use_deg_estimator = use_deg_estimator
        self.use_spatial_attn = use_spatial_attn
        self.use_film = use_film
        self.use_text_code = use_text_code
        if text_features_all is not None:
            self.register_buffer('text_features_all', text_features_all)

        self.patch_embed = OverlapPatchEmbed(inp_channels, dim)
        self.encoder_level1 = nn.Sequential(*[TransformerBlock(dim=dim, num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])
        self.encoder_shuffle_channel1 = ch_shuffle_high_text(ch_dim=dim, num_heads=heads[0], LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.down1_2 = Downsample(dim)
        self.encoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])
        self.encoder_shuffle_channel2 = ch_shuffle_high_text(ch_dim=int(dim*2**1), num_heads=heads[1], LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.down2_3 = Downsample(int(dim*2**1))
        self.encoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])
        self.encoder_shuffle_channel3 = ch_shuffle_high_text(ch_dim=int(dim*2**2), num_heads=heads[2], LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.down3_4 = Downsample(int(dim*2**2))
        self.latent = nn.Sequential(*[TransformerBlock(dim=int(dim*2**3), num_heads=heads[3], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[3])])
        self.latent_shuffle_channel = ch_shuffle_high_text(ch_dim=int(dim*2**3), num_heads=heads[3], LayerNorm_type=LayerNorm_type, ffn_expansion_factor=ffn_expansion_factor, bias=bias)

        self.up4_3 = Upsample(int(dim*2**3))
        self.reduce_chan_level3 = nn.Conv2d(int(dim*2**3), int(dim*2**2), kernel_size=1, bias=bias)
        self.decoder_level3 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**2), num_heads=heads[2], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[2])])

        self.up3_2 = Upsample(int(dim*2**2))
        self.reduce_chan_level2 = nn.Conv2d(int(dim*2**2), int(dim*2**1), kernel_size=1, bias=bias)
        self.decoder_level2 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[1], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[1])])

        self.up2_1 = Upsample(int(dim*2**1))
        self.decoder_level1 = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_blocks[0])])

        self.refinement = nn.Sequential(*[TransformerBlock(dim=int(dim*2**1), num_heads=heads[0], ffn_expansion_factor=ffn_expansion_factor, bias=bias, LayerNorm_type=LayerNorm_type) for i in range(num_refinement_blocks)])

        self.output = nn.Conv2d(int(dim*2**1), out_channels, kernel_size=3, stride=1, padding=1, bias=bias)

        self.deg_estimator = DegradationEstimator(num_types=num_types)

        self.spatial_attn3 = SeveritySpatialAttention(int(dim * 2**2))
        self.spatial_attn2 = SeveritySpatialAttention(int(dim * 2**1))
        self.spatial_attn1 = SeveritySpatialAttention(dim)

        self.film3 = FiLM(int(dim * 2**2))
        self.film2 = FiLM(int(dim * 2**1))
        self.film1 = FiLM(int(dim * 2**1))

    def forward(self, inp_img):
        b = inp_img.shape[0]
        if self.use_deg_estimator:
            severity_map, type_logits = self.deg_estimator(inp_img)      # (B,1,H,W), (B,7)
            severity_scalar = severity_map.mean(dim=[2, 3])             # (B, 1)
            type_probs = torch.sigmoid(type_logits)                     # (B,7) probs for text blend
            text_code = type_probs @ self.text_features_all             # (B, 512)
        else:
            # w/o DegEstimator: no per-image severity or type signal at all —
            # fall back to a fixed prompt (mean of the 7 CLIP task embeddings)
            # and a flat severity map/scalar so downstream modules still run
            # structurally but receive no image-adaptive information.
            severity_map = torch.zeros(b, 1, inp_img.shape[2], inp_img.shape[3], device=inp_img.device, dtype=inp_img.dtype)
            severity_scalar = torch.zeros(b, 1, device=inp_img.device, dtype=inp_img.dtype)
            type_logits = torch.zeros(b, self.text_features_all.shape[0], device=inp_img.device, dtype=inp_img.dtype)
            text_code = self.text_features_all.mean(dim=0, keepdim=True).expand(b, -1)

        inp_enc_level1 = self.patch_embed(inp_img)
        out_enc_level1 = self.encoder_level1(inp_enc_level1)

        inp_enc_level2 = self.down1_2(out_enc_level1)
        out_enc_level2 = self.encoder_level2(inp_enc_level2)

        inp_enc_level3 = self.down2_3(out_enc_level2)
        out_enc_level3 = self.encoder_level3(inp_enc_level3)

        inp_enc_level4 = self.down3_4(out_enc_level3)
        latent = self.latent(inp_enc_level4)
        latent, _ = self.latent_shuffle_channel(latent, text_code)

        inp_dec_level3 = self.up4_3(latent)
        outt1, _ = self.encoder_shuffle_channel3(out_enc_level3, text_code)
        if self.use_spatial_attn:
            outt1 = self.spatial_attn3(outt1, severity_map)
        inp_dec_level3 = self.reduce_chan_level3(torch.cat([inp_dec_level3, outt1], 1))
        out_dec_level3 = self.decoder_level3(inp_dec_level3)
        if self.use_film:
            out_dec_level3 = self.film3(out_dec_level3, severity_scalar)

        inp_dec_level2 = self.up3_2(out_dec_level3)
        outt2, _ = self.encoder_shuffle_channel2(out_enc_level2, text_code)
        if self.use_spatial_attn:
            outt2 = self.spatial_attn2(outt2, severity_map)
        inp_dec_level2 = self.reduce_chan_level2(torch.cat([inp_dec_level2, outt2], 1))
        out_dec_level2 = self.decoder_level2(inp_dec_level2)
        if self.use_film:
            out_dec_level2 = self.film2(out_dec_level2, severity_scalar)


        inp_dec_level1 = self.up2_1(out_dec_level2)
        outt3, _ = self.encoder_shuffle_channel1(out_enc_level1, text_code)
        if self.use_spatial_attn:
            outt3 = self.spatial_attn1(outt3, severity_map)
        inp_dec_level1 = torch.cat([inp_dec_level1, outt3], 1)
        out_dec_level1 = self.decoder_level1(inp_dec_level1)
        if self.use_film:
            out_dec_level1 = self.film1(out_dec_level1, severity_scalar)

        out_dec_level1 = self.refinement(out_dec_level1)
        out_dec_level1 = self.output(out_dec_level1) + inp_img
  
        return torch.clamp(out_dec_level1, 0.0, 1.0), severity_map, type_logits
