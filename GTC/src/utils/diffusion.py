from tqdm import tqdm
from utils import *
import logging
import torch
from utils.unet import UNet_conditional


logging.basicConfig(format="%(asctime)s - %(levelname)s: %(message)s", level=logging.INFO, datefmt="%I:%M:%S")


class Diffusion:
    def __init__(self, noise_steps=100, beta_start=1e-4, beta_end=0.02, device="cuda"):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        # self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule().to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        self.denosing = UNet_conditional(embed_dim=64, time_dim=64, cond_dim=64).to(device)

    def prepare_noise_schedule(self):
        return torch.linspace(self.beta_start, self.beta_end, self.noise_steps)

    def noise_images(self, x, t, cond):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None]
        
        # Generate pure Gaussian noise
        noise = torch.randn_like(x)
        
        # Forward diffusion: x_t = sqrt(alpha_hat_t) * x_0 + sqrt(1 - alpha_hat_t) * epsilon
        noised_x = sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * noise
        return noised_x, noise

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))

    def denoisesample(self, x, cond):
        """
          x_{t-1} = 1/sqrt(alpha_t) * (x_t - ((1 - alpha_t) / sqrt(1 - alpha_hat_t)) * predicted_noise) + sqrt(beta_t) * z
        """
        for t in reversed(range(self.noise_steps)):
            time_tensor = torch.full((x.shape[0],), t, device=self.device, dtype=torch.long)
            predicted_noise = self.denosing(x, time_tensor, cond)
            beta_t = self.beta[t]
            alpha_t = self.alpha[t]
            alpha_hat_t = self.alpha_hat[t]
            if t > 0:
                z = torch.randn_like(x)
            else:
                z = 0
            # Reverse diffusion step
            x = (1 / torch.sqrt(alpha_t)) * (x - ((1 - alpha_t) / torch.sqrt(1 - alpha_hat_t)) * predicted_noise) + torch.sqrt(beta_t) * z
        return x, predicted_noise