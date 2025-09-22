import torch
import torchaudio
import torch.nn.functional as F
import numpy as np

def scale_and_clamp_waveform(enhanced_wav, original_peak):
    denormalized_wav = enhanced_wav * original_peak
    max_val = torch.max(torch.abs(denormalized_wav))
    if max_val > 1.0:
        print(f"Clipping detected (max value: {max_val:.4f}). Normalizing peak to 1.0.")
        final_wav = denormalized_wav / max_val
    else:
        final_wav = denormalized_wav
    return final_wav

def extract_energy_envelope(waveform, win_size, hop_length):
    if waveform.dim() > 1:
        waveform = torch.mean(waveform, dim=0)
    energy = waveform.pow(2)
    
    energy_with_dims = energy.unsqueeze(0).unsqueeze(0)
    
    envelope = F.conv1d(
        energy_with_dims,
        torch.ones(1, 1, win_size, device=waveform.device) / win_size,
        stride=hop_length,
        padding=(win_size - 1) // 2
    ).squeeze(0).squeeze(0)
    
    return torch.log(envelope + 1e-8)

def to_spectrogram(waveform, n_fft, hop_length, win_size):
    window = torch.hann_window(win_size, device=waveform.device)
    spec = torch.stft(
        waveform,
        n_fft=n_fft,
        hop_length=hop_length,
        win_length=win_size,
        window=window,
        return_complex=True
    )
    return spec

def to_waveform(spectrogram, n_fft, hop_length):
    win_length = n_fft
    window = torch.hann_window(win_length, device=spectrogram.device)
    waveform = torch.istft(
        spectrogram, 
        n_fft=n_fft, 
        hop_length=hop_length, 
        win_length=win_length, 
        window=window
    )
    return waveform

def save_audio(waveform, path, sample_rate):
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    torchaudio.save(path, waveform.cpu(), sample_rate)

def calculate_ssnr(clean_speech, processed_speech, frame_len=256, hop_len=128):
    clean_speech = clean_speech - np.mean(clean_speech)
    processed_speech = processed_speech - np.mean(processed_speech)
    num_frames = (len(clean_speech) - frame_len) // hop_len + 1
    if num_frames <= 0: return 0.0
    snr_list = []
    for i in range(num_frames):
        start = i * hop_len
        end = start + frame_len
        clean_frame = clean_speech[start:end]
        processed_frame = processed_speech[start:end]
        signal_power = np.sum(clean_frame**2)
        noise_power = np.sum((clean_frame - processed_frame)**2)
        if signal_power > 1e-8 and noise_power > 1e-8:
            snr = 10 * np.log10(signal_power / noise_power)
            snr_list.append(snr)
    if not snr_list: return 0.0
    return np.mean(snr_list)