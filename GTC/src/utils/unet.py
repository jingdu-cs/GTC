
import torch
import torch.nn as nn


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class DoubleLinear(nn.Module):
    """Double fully connected block with two linear layers followed by activation."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.double_linear = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(inplace=True),
            nn.Linear(out_features, out_features),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_linear(x)

class DownEmbedding(nn.Module):
    """Down-sampling block for embedding space using linear layers."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.down = DoubleLinear(in_features, out_features)

    def forward(self, x):
        return self.down(x)

class UpEmbedding(nn.Module):
    """Up-sampling block for embedding space using linear layers."""
    def __init__(self, in_features, out_features):
        super().__init__()
        self.up = nn.Linear(in_features, out_features)
        self.conv = DoubleLinear(out_features * 2, out_features)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=-1)  # Concatenate along the feature dimension
        return self.conv(x)

class UNet(nn.Module):
    """UNet for processing image embeddings directly."""
    def __init__(self, embed_dim, time_dim=64, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        # self.hidden_dim=64

        # Initial block for embedding input
        self.embedding_fc = DoubleLinear(embed_dim, 64)

        # Encoder
        self.down1 = DownEmbedding(64, 128)
        self.down2 = DownEmbedding(128, 256)
        self.down3 = DownEmbedding(256, 256)

        # Bottleneck
        self.bot1 = DoubleLinear(256, 512)
        self.bot2 = DoubleLinear(512, 512)
        self.bot3 = DoubleLinear(512, 256)

        # Decoder
        self.up1 = UpEmbedding(256, 256)
        self.up2 = UpEmbedding(256, 128)
        self.up3 = UpEmbedding(128, 64)

        # Output layer
        self.out_fc = nn.Linear(64, embed_dim)

    def pos_encoding(self, t, channels):
        """Positional encoding for time-dependent features."""
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t):
        """Forward pass for embedding input."""
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)

        # Initial embedding block
        x1 = self.embedding_fc(x+t)

        # Encoder path
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        # Bottleneck
        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        # Decoder path
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)

        # Final output
        return self.out_fc(x)


class UNet_conditional(nn.Module):
    """针对嵌入向量的 UNet 去噪网络"""
    def __init__(self, embed_dim, time_dim=64, cond_dim=32, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = embed_dim + cond_dim
        self.cond_dim = cond_dim

        # 初始嵌入层
        self.embedding_fc = DoubleLinear(embed_dim + cond_dim, 64)

        # Encoder
        self.down1 = DownEmbedding(64, 128)
        self.down2 = DownEmbedding(128, 128)

        # Bottleneck
        self.bot1 = DoubleLinear(128, 256)
        self.bot2 = DoubleLinear(256, 128)

        # Decoder
        self.up1 = UpEmbedding(128, 128)
        self.up2 = UpEmbedding(128, 64)

        # 输出层
        self.out_fc = nn.Linear(64, embed_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t, cond):
        # 时间嵌入
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        # Concatenate condition with input
        x = torch.cat([x, cond], dim=-1)
        
        x1 = self.embedding_fc(x + t)
        
        # Encoder 路径
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)

        # Decoder 路径
        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        # 输出
        return self.out_fc(x)




# class UNet_conditional(nn.Module):
#     def __init__(self, c_in=3, c_out=3, time_dim=256, num_classes=None, device="cuda"):
#         super().__init__()
#         self.device = device
#         self.time_dim = time_dim
#         self.inc = DoubleConv(c_in, 64)
#         self.down1 = Down(64, 128)
#         self.sa1 = SelfAttention(128, 32)
#         self.down2 = Down(128, 256)
#         self.sa2 = SelfAttention(256, 16)
#         self.down3 = Down(256, 256)
#         self.sa3 = SelfAttention(256, 8)
#
#         self.bot1 = DoubleConv(256, 512)
#         self.bot2 = DoubleConv(512, 512)
#         self.bot3 = DoubleConv(512, 256)
#
#         self.up1 = Up(512, 128)
#         self.sa4 = SelfAttention(128, 16)
#         self.up2 = Up(256, 64)
#         self.sa5 = SelfAttention(64, 32)
#         self.up3 = Up(128, 64)
#         self.sa6 = SelfAttention(64, 64)
#         self.outc = nn.Conv2d(64, c_out, kernel_size=1)
#
#         if num_classes is not None:
#             self.label_emb = nn.Embedding(num_classes, time_dim)
#
#     def pos_encoding(self, t, channels):
#         inv_freq = 1.0 / (
#             10000
#             ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
#         )
#         pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
#         pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
#         return pos_enc
#
#     def forward(self, x, t, y):
#         t = t.unsqueeze(-1).type(torch.float)
#         t = self.pos_encoding(t, self.time_dim)
#
#         if y is not None:
#             t += self.label_emb(y)
#
#         x1 = self.inc(x)
#         x2 = self.down1(x1, t)
#         x2 = self.sa1(x2)
#         x3 = self.down2(x2, t)
#         x3 = self.sa2(x3)
#         x4 = self.down3(x3, t)
#         x4 = self.sa3(x4)
#
#         x4 = self.bot1(x4)
#         x4 = self.bot2(x4)
#         x4 = self.bot3(x4)
#
#         x = self.up1(x4, x3, t)
#         x = self.sa4(x)
#         x = self.up2(x, x2, t)
#         x = self.sa5(x)
#         x = self.up3(x, x1, t)
#         x = self.sa6(x)
#         output = self.outc(x)
#         return output
