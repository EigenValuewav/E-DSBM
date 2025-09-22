import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import torchaudio.transforms as T
import os
import random
from utils import extract_energy_envelope

def to_spectrogram(wav, n_fft, hop_length, win_size):
    win = torch.hann_window(win_size, device=wav.device)
    spec = torch.stft(wav, n_fft=n_fft, hop_length=hop_length, win_length=win_size, window=win, return_complex=True)
    return spec

def get_dataset_filelist(train_file, test_file):
    with open(train_file, 'r', encoding='utf-8') as fi:
        training_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    with open(test_file, 'r', encoding='utf-8') as fi:
        validation_indexes = [x.split('|')[0] for x in fi.read().split('\n') if len(x) > 0]

    return training_indexes, validation_indexes

class DSBMSpeechDataset(torch.utils.data.Dataset):
    def __init__(self, file_indexes, clean_dir, noisy_dir, segment_size, 
                 sampling_rate, n_fft, hop_size, win_size, shuffle=True):
        self.audio_indexes = file_indexes
        if shuffle:
            random.seed(1234)
            random.shuffle(self.audio_indexes)
            
        self.clean_dir = clean_dir
        self.noisy_dir = noisy_dir
        self.segment_size = segment_size
        self.sampling_rate = sampling_rate
        
        self.n_fft = n_fft
        self.hop_size = hop_size
        self.win_size = win_size
        
        self.resampler = {}

    def __getitem__(self, index):
        filename = self.audio_indexes[index]

        clean_path = os.path.join(self.clean_dir, filename + '.wav')
        noisy_path = os.path.join(self.noisy_dir, filename + '.wav')

        clean_wav, sr_c = torchaudio.load(clean_path)
        noisy_wav, sr_n = torchaudio.load(noisy_path)
        
        if sr_c != self.sampling_rate:
            if sr_c not in self.resampler:
                self.resampler[sr_c] = T.Resample(sr_c, self.sampling_rate).to(clean_wav.device)
            clean_wav = self.resampler[sr_c](clean_wav)

        if sr_n != self.sampling_rate:
            if sr_n not in self.resampler:
                self.resampler[sr_n] = T.Resample(sr_n, self.sampling_rate).to(noisy_wav.device)
            noisy_wav = self.resampler[sr_n](noisy_wav)

        length = min(clean_wav.size(1), noisy_wav.size(1))
        clean_wav, noisy_wav = clean_wav[:, :length], noisy_wav[:, :length]

        if length < self.segment_size:
            pad_len = self.segment_size - length
            clean_segment = F.pad(clean_wav, (0, pad_len), 'constant')
            noisy_segment = F.pad(noisy_wav, (0, pad_len), 'constant')
        else:
            max_start = length - self.segment_size
            start = random.randint(0, max_start)
            clean_segment = clean_wav[:, start : start + self.segment_size]
            noisy_segment = noisy_wav[:, start : start + self.segment_size]
        
        clean_segment_1d = clean_segment.squeeze(0)
        noisy_segment_1d = noisy_segment.squeeze(0)

        clean_envelope = extract_energy_envelope(clean_segment_1d, win_size=self.win_size, hop_length=self.hop_size)
        noisy_envelope = extract_energy_envelope(noisy_segment_1d, win_size=self.win_size, hop_length=self.hop_size)

        clean_spec_complex = to_spectrogram(clean_segment_1d, n_fft=self.n_fft, hop_length=self.hop_size, win_size=self.win_size)
        noisy_spec_complex = to_spectrogram(noisy_segment_1d, n_fft=self.n_fft, hop_length=self.hop_size, win_size=self.win_size)

        clean_spec_ri = torch.view_as_real(clean_spec_complex).permute(2, 0, 1)
        noisy_spec_ri = torch.view_as_real(noisy_spec_complex).permute(2, 0, 1)

        return {
            "clean_spec": clean_spec_ri, 
            "noisy_spec": noisy_spec_ri, 
            "clean_envelope": clean_envelope,
            "noisy_envelope": noisy_envelope,
            "clean_wav": clean_segment,
            "noisy_wav": noisy_segment
        }
        
    def __len__(self):
        return len(self.audio_indexes)