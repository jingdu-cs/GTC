
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
    def __init__(self, embed_dim, time_dim=64, cond_dim=32, device="cuda"):
        super().__init__()
        self.device = device
        self.time_dim = embed_dim + cond_dim
        self.cond_dim = cond_dim

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

        self.out_fc = nn.Linear(64, embed_dim)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2, device=self.device).float() / channels))
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        return torch.cat([pos_enc_a, pos_enc_b], dim=-1)

    def forward(self, x, t, cond):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        
        # Concatenate condition with input
        x = torch.cat([x, cond], dim=-1)
        
        x1 = self.embedding_fc(x + t)
        
        # Encoder
        x2 = self.down1(x1)
        x3 = self.down2(x2)

        # Bottleneck
        x3 = self.bot1(x3)
        x3 = self.bot2(x3)

        # Decoder
        x = self.up1(x3, x2)
        x = self.up2(x, x1)

        return self.out_fc(x)



