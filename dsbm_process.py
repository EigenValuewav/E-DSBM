import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import to_waveform

# Multi Resolution STFT Loss
class MultiResolutionSTFTLoss(nn.Module):
    def __init__(self, fft_sizes=[1024, 2048, 512], hop_sizes=[120, 240, 50], win_lengths=[600, 1200, 240], window="hann_window"):
        super(MultiResolutionSTFTLoss, self).__init__()
        self.fft_sizes = fft_sizes
        self.hop_sizes = hop_sizes
        self.win_lengths = win_lengths
        self.window = getattr(torch, window, False)

    def forward(self, x, y):
        sc_loss = 0.0
        mag_loss = 0.0 
        
        for fft, hop, win in zip(self.fft_sizes, self.hop_sizes, self.win_lengths):
            x_spec = torch.stft(x.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=self.window(win, device=x.device), return_complex=True, center=True)
            y_spec = torch.stft(y.squeeze(1), n_fft=fft, hop_length=hop, win_length=win, window=self.window(win, device=y.device), return_complex=True, center=True)
            
            x_mag = torch.sqrt(x_spec.real**2 + x_spec.imag**2 + 1e-7)
            y_mag = torch.sqrt(y_spec.real**2 + y_spec.imag**2 + 1e-7)

            sc_loss += torch.norm(y_mag - x_mag, p="fro") / torch.norm(y_mag, p="fro")
            mag_loss += F.l1_loss(torch.log(y_mag), torch.log(x_mag))
            
        return sc_loss + mag_loss

# SI-SNR Loss
def si_snr_loss(preds, target, epsilon=1e-8):
    if preds.dim() > 1 and preds.shape[1] == 1:
        preds = preds.squeeze(1)
    if target.dim() > 1 and target.shape[1] == 1:
        target = target.squeeze(1)

    preds = preds - torch.mean(preds, dim=1, keepdim=True)
    target = target - torch.mean(target, dim=1, keepdim=True)

    s_target = torch.sum(preds * target, dim=1, keepdim=True) * target / (torch.sum(target**2, dim=1, keepdim=True) + epsilon)
    e_noise = preds - s_target

    snr = torch.sum(s_target**2, dim=1) / (torch.sum(e_noise**2, dim=1) + epsilon)
    snr_db = 10 * torch.log10(snr + epsilon)
    
    return -torch.mean(snr_db)


class DSBM_Enhancer:
    def __init__(self, model, config, sigma=0.5, T=1.0, 
                 stft_loss_weight=1.0, 
                 sisnr_loss_weight=1.0, 
                 envelope_loss_weight=0.1):
        self.model = model
        self.config = config
        self.sigma = sigma
        self.T = T
        self.stft_loss_fn = MultiResolutionSTFTLoss().to(next(model.parameters()).device)
        self.stft_loss_weight = stft_loss_weight
        self.sisnr_loss_weight = sisnr_loss_weight
        self.envelope_loss_weight = envelope_loss_weight

    def calculate_loss(self, clean_spec, noisy_spec, clean_envelope, noisy_envelope, clean_wav, noisy_wav):
        device = clean_spec.device
        t = torch.rand(clean_spec.shape[0], device=device) * self.T
        
        Z = torch.randn_like(clean_spec)
        sigma_t = self.sigma * torch.sqrt(t * (1.0 - t / self.T))
        
        t_view = t.view(-1, 1, 1, 1)
        sigma_t_view = sigma_t.view(-1, 1, 1, 1)
        z_t = (1.0 - t_view / self.T) * noisy_spec + (t_view / self.T) * clean_spec + sigma_t_view * Z

        predicted_drift, predicted_envelope = self.model(z_t, t, x_cond=noisy_spec, envelope=noisy_envelope)
        predicted_clean_spec = noisy_spec + predicted_drift
        predicted_clean_complex = torch.view_as_complex(predicted_clean_spec.permute(0, 2, 3, 1).contiguous())
        
        predicted_clean_wav = to_waveform(
            predicted_clean_complex, 
            n_fft=self.config.audio.n_fft, 
            hop_length=self.config.audio.hop_size
        )

        loss_stft = self.stft_loss_fn(predicted_clean_wav, clean_wav)
        loss_sisnr = si_snr_loss(predicted_clean_wav, clean_wav)
        
        target_len = clean_envelope.shape[-1]
        predicted_envelope_resized = F.interpolate(
            predicted_envelope.unsqueeze(1), size=target_len, mode='linear', align_corners=False
        ).squeeze(1)
        loss_envelope = F.mse_loss(predicted_envelope_resized, clean_envelope)

        total_loss = (self.stft_loss_weight * loss_stft + 
                      self.sisnr_loss_weight * loss_sisnr + 
                      self.envelope_loss_weight * loss_envelope)
        
        return total_loss, loss_stft, loss_sisnr, loss_envelope
    
    @torch.no_grad()
    # infer. sampling
    def sample(self, noisy_spec, noisy_envelope, num_steps=100):
        device = noisy_spec.device
        dt = self.T / num_steps
        z = noisy_spec.clone()
        
        for i in range(num_steps):
            t_val_scalar = i * dt
            batch_size = z.shape[0]
            t_val_tensor = torch.full((batch_size,), t_val_scalar, device=device, dtype=torch.float32)
            drift, _ = self.model(z, t_val_tensor, x_cond=noisy_spec, envelope=noisy_envelope)
            z = z + drift * dt
            
        return z
    