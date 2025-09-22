import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
import os
import argparse
import numpy as np
import torch.nn.functional as F
from env import AttrDict
from dataset import DSBMSpeechDataset, get_dataset_filelist
from unet import ConditionalUnet
from dsbm_process import DSBM_Enhancer
from utils import to_waveform, save_audio, to_spectrogram, extract_energy_envelope, calculate_ssnr, scale_and_clamp_waveform
from torch.utils.tensorboard import SummaryWriter
import json
import torchaudio
import torchaudio.transforms as T
import pandas as pd
from pesq import pesq
from pystoi import stoi

torch.backends.cudnn.benchmark = True

def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        json_config = json.load(f)
    return AttrDict(json_config)

def train(args, h):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    writer = SummaryWriter(args.log_dir)
    os.makedirs(args.save_dir, exist_ok=True)
    print(f"Checkpoints will be saved in: {args.save_dir}")
    training_indexes, validation_indexes = get_dataset_filelist(
        train_file=h.data.train_file, test_file=h.data.test_file
    )
    dataset = DSBMSpeechDataset(
        file_indexes=training_indexes,
        clean_dir=h.data.clean_dir,
        noisy_dir=h.data.noisy_dir,
        segment_size=h.audio.segment_size,
        sampling_rate=h.audio.sampling_rate,
        n_fft=h.audio.n_fft,
        hop_size=h.audio.hop_size,
        win_size=h.audio.win_size,
        shuffle=True
    )
    dataloader = DataLoader(dataset, batch_size=h.train.batch_size, shuffle=True, num_workers=h.train.num_workers, pin_memory=True)

    val_dataset = DSBMSpeechDataset(
        file_indexes=validation_indexes,
        clean_dir=h.data.clean_dir,
        noisy_dir=h.data.noisy_dir,
        segment_size=h.audio.segment_size,
        sampling_rate=h.audio.sampling_rate,
        n_fft=h.audio.n_fft,
        hop_size=h.audio.hop_size,
        win_size=h.audio.win_size,
        shuffle=False
    )
    val_loader = DataLoader(val_dataset, batch_size=h.train.batch_size, shuffle=False, num_workers=h.train.num_workers, pin_memory=True)

    model = ConditionalUnet(
        dim=h.model.dim,
        dim_mults=tuple(h.model.dim_mults),
        channels=h.model.channels,
        self_condition=h.model.self_condition,
        envelope_emb_dim=h.model.envelope_emb_dim
    ).to(device)
    
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total number of trainable parameters: {num_params:,}")
    
    # --finetune finetune option (config or arg)
    learning_rate = args.ft_lr if args.finetune else h.train.learning_rate
    if args.finetune:
        print(f"Fine-tuning mode enabled. Using learning rate: {learning_rate}")
    
    optimizer = Adam(model.parameters(), lr=learning_rate, betas=(h.train.adam_b1, h.train.adam_b2))
    
    dsbm_enhancer = DSBM_Enhancer(model, config=h,
                                  sigma=h.train.sigma,
                                  stft_loss_weight=h.loss_weights.stft,
                                  sisnr_loss_weight=h.loss_weights.sisnr,
                                  envelope_loss_weight=h.loss_weights.envelope)
    scaler = GradScaler()
    start_epoch = 0
    best_val_loss = float('inf')

    if args.resume_checkpoint and os.path.isfile(args.resume_checkpoint):
        print(f"Resuming from checkpoint: {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=False)
        print("Model state loaded with strict=False.")
        
        if 'optimizer_state_dict' in checkpoint and not args.finetune:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            print("Optimizer state loaded for continued training.")
        elif args.finetune:
            print("Skipping optimizer state loading for fine-tuning.")

        start_epoch = checkpoint.get('epoch', 0)
        best_val_loss = checkpoint.get('val_loss', float('inf'))
        print(f"Resuming from Epoch {start_epoch + 1}")

    print("Starting Training...")
    
    # --finetune epoch option (config or arg)
    total_epochs = args.epochs if args.epochs is not None else h.train.epochs
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.epochs} [Train]")
        total_loss, total_stft_loss, total_sisnr_loss, total_env_loss = 0, 0, 0, 0
        
        for batch in pbar:
            noisy_spec = batch['noisy_spec'].to(device)
            clean_spec = batch['clean_spec'].to(device)
            noisy_envelope = batch['noisy_envelope'].to(device)
            clean_envelope = batch['clean_envelope'].to(device)
            noisy_wav = batch['noisy_wav'].to(device)
            clean_wav = batch['clean_wav'].to(device)

            optimizer.zero_grad()
            with autocast():
                loss, loss_stft, loss_sisnr, loss_env = dsbm_enhancer.calculate_loss(
                    clean_spec, noisy_spec, clean_envelope, noisy_envelope, clean_wav, noisy_wav
                )
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            # Loss Calculate
            total_loss += loss.item()
            total_stft_loss += loss_stft.item()
            total_sisnr_loss += loss_sisnr.item()
            total_env_loss += loss_env.item()
            
            pbar.set_postfix(loss=loss.item(), stft=loss_stft.item(), sisnr=loss_sisnr.item(), env=loss_env.item())
        
        avg_loss = total_loss / len(dataloader)
        avg_stft_loss = total_stft_loss / len(dataloader)
        avg_sisnr_loss = total_sisnr_loss / len(dataloader)
        avg_env_loss = total_env_loss / len(dataloader)
        print(f"Epoch {epoch+1} Avg Train Losses -> Total: {avg_loss:.4f}, STFT: {avg_stft_loss:.4f}, SI-SNR: {avg_sisnr_loss:.4f}, Env: {avg_env_loss:.4f}")

        writer.add_scalar('Loss/train_total', avg_loss, epoch + 1)
        writer.add_scalar('Loss/train_stft', avg_stft_loss, epoch + 1)
        writer.add_scalar('Loss/train_sisnr', avg_sisnr_loss, epoch + 1)
        writer.add_scalar('Loss/train_envelope', avg_env_loss, epoch + 1)

        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in tqdm(val_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Val]"):
                noisy_spec = batch['noisy_spec'].to(device)
                clean_spec = batch['clean_spec'].to(device)
                noisy_envelope = batch['noisy_envelope'].to(device)
                clean_envelope = batch['clean_envelope'].to(device)
                noisy_wav = batch['noisy_wav'].to(device)
                clean_wav = batch['clean_wav'].to(device)

                loss, _, _, _ = dsbm_enhancer.calculate_loss(
                    clean_spec, noisy_spec, clean_envelope, noisy_envelope, clean_wav, noisy_wav
                )
                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        print(f"Epoch {epoch+1} Avg Val Loss -> Total: {avg_val_loss:.4f}")
        
        writer.add_scalar('Loss/val_total', avg_val_loss, epoch + 1)
        writer.add_scalar('Misc/learning_rate', optimizer.param_groups[0]['lr'], epoch + 1)

        if (epoch + 1) % h.train.save_interval == 0:
            save_path = os.path.join(args.save_dir, f"model_epoch_{epoch+1}.pth")
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"Checkpoint saved to {save_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = os.path.join(args.save_dir, "best_model.pth")
            torch.save({
                'epoch': epoch + 1, 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': best_val_loss,
            }, save_path)
            print(f"🎉 New best model found! Checkpoint saved to {save_path}")

    writer.close()
    print("Training finished.")

def inference_and_evaluate(args, h, mode='inference'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device} for {mode}")

    model = ConditionalUnet(
        dim=h.model.dim,
        dim_mults=tuple(h.model.dim_mults),
        channels=h.model.channels,
        self_condition=h.model.self_condition,
        envelope_emb_dim=h.model.envelope_emb_dim
    ).to(device)
    
    print(f"Loading checkpoint from: {args.checkpoint_path}")
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint.get('model_state_dict', checkpoint))
    print(f"Checkpoint from epoch {checkpoint.get('epoch', 'N/A')} loaded successfully.")
    model.eval()

    dsbm_enhancer = DSBM_Enhancer(model, h, sigma=h.train.sigma)

    if mode == 'inference':
        process_single_file(args, h, dsbm_enhancer, device)
    elif mode == 'evaluate':
        process_evaluation_set(args, h, dsbm_enhancer, device)

def process_single_file(args, h, dsbm_enhancer, device, input_path=None, output_path=None):
    input_path = input_path or args.input_path
    output_path = output_path or args.output_path
    
    if output_path is None:
        input_filename = os.path.basename(input_path)
        os.makedirs('output', exist_ok=True)
        output_path = os.path.join('output', f"enhanced_{input_filename}")
        print(f"Output path not specified. Saving to: {output_path}")

    noisy_wav, sr = torchaudio.load(input_path)
    if noisy_wav.dim() == 1:
        noisy_wav = noisy_wav.unsqueeze(0)
    noisy_wav = noisy_wav.to(device)
    
    if noisy_wav.shape[0] > 1:
        noisy_wav = torch.mean(noisy_wav, dim=0, keepdim=True)
    if sr != h.audio.sampling_rate:
        resampler = T.Resample(sr, h.audio.sampling_rate).to(device)
        noisy_wav = resampler(noisy_wav)

    noisy_envelope = extract_energy_envelope(noisy_wav.squeeze(0), win_size=h.audio.win_size, hop_length=h.audio.hop_size)
    original_peak = torch.max(torch.abs(noisy_wav))
    normalized_noisy_wav = noisy_wav / (original_peak + 1e-8)
    noisy_spec_complex = to_spectrogram(normalized_noisy_wav.squeeze(0), n_fft=h.audio.n_fft, hop_length=h.audio.hop_size, win_size=h.audio.win_size)
    noisy_spec_ri = torch.view_as_real(noisy_spec_complex).permute(2, 0, 1).unsqueeze(0)
    
    with torch.no_grad():
        enhanced_spec_ri = dsbm_enhancer.sample(noisy_spec_ri, noisy_envelope.unsqueeze(0), num_steps=args.inference_steps)
    
    result_spec = enhanced_spec_ri.squeeze(0).permute(1, 2, 0).contiguous()
    enhanced_spec_complex = torch.view_as_complex(result_spec)
    enhanced_wav = to_waveform(enhanced_spec_complex, n_fft=h.audio.n_fft, hop_length=h.audio.hop_size)
    final_enhanced_wav = scale_and_clamp_waveform(enhanced_wav, original_peak)
    
    if output_path:
        save_audio(final_enhanced_wav.cpu(), output_path, sample_rate=h.audio.sampling_rate)

    return final_enhanced_wav.cpu()


def process_evaluation_set(args, h, dsbm_enhancer, device):
    os.makedirs(args.results_dir, exist_ok=True)
    test_files = [f for f in os.listdir(args.test_noisy_dir) if f.endswith(('.wav', '.flac'))]
    results = []

    for filename in tqdm(test_files, desc="Evaluating Test Set"):
        try:
            noisy_path = os.path.join(args.test_noisy_dir, filename)
            clean_path = os.path.join(args.test_clean_dir, filename)
            output_path = os.path.join(args.results_dir, f"{os.path.splitext(filename)[0]}_enhanced.wav")

            enhanced_wav_eval_cpu = process_single_file(args, h, dsbm_enhancer, device, input_path=noisy_path, output_path=output_path)
            
            clean_wav_eval, _ = torchaudio.load(clean_path)

            min_len = min(clean_wav_eval.shape[1], enhanced_wav_eval_cpu.shape[1])
            clean_np = clean_wav_eval[0, :min_len].numpy()
            enhanced_np = enhanced_wav_eval_cpu[0, :min_len].numpy()
            
            pesq_score = np.nan
            if h.audio.sampling_rate in [8000, 16000]:
                pesq_score = pesq(h.audio.sampling_rate, clean_np, enhanced_np, 'wb' if h.audio.sampling_rate == 16000 else 'nb')
            
            stoi_score = stoi(clean_np, enhanced_np, h.audio.sampling_rate, extended=False)
            ssnr_score = calculate_ssnr(clean_np, enhanced_np)

            results.append({"filename": filename, "PESQ": pesq_score, "STOI": stoi_score, "SSNR": ssnr_score})
        except Exception as e:
            print(f"\n[Warning] Could not process file {filename}. Error: {e}")
            continue
    
    if results:
        df = pd.DataFrame(results)
        df.to_csv(args.results_csv, index=False)
        print(f"\nEvaluation finished. Results saved to {args.results_csv}")
        print("\n--- Average Scores ---")
        print(df.mean(numeric_only=True))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # train args.
    parser.add_argument('--config', type=str, default='config.json', help='Path to configuration json file.')
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference', 'evaluate'])
    parser.add_argument('--checkpoint_path', type=str, help='Path to checkpoint for inference or evaluation.')
    parser.add_argument('--epochs', type=int, help='Override number of epochs in config.')
    parser.add_argument('--log_dir', type=str, default='runs', help="Directory for TensorBoard logs")
    parser.add_argument('--save_dir', type=str, default='checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--resume_checkpoint', type=str, help="Path to checkpoint to resume training from")
    parser.add_argument('--finetune', action='store_true', help="Enable fine-tuning mode.")
    parser.add_argument('--ft_lr', type=float, help="Learning rate for fine-tuning. Overrides config.")
    
    # Inference & Evaluate args.
    parser.add_argument('--input_path', type=str, help='Path to input audio for inference.')
    parser.add_argument('--output_path', type=str, help='Path to save enhanced audio.')
    parser.add_argument('--inference_steps', type=int, default=10)
    parser.add_argument('--test_noisy_dir', type=str, help="Directory of noisy wav files for evaluation", default="./VoiceBank+DEMAND/testset_noisy")
    parser.add_argument('--test_clean_dir', type=str, help="Directory of clean wav files for evaluation", default="./VoiceBank+DEMAND/testset_clean")
    parser.add_argument('--results_dir', type=str, default='./eval_results')
    parser.add_argument('--results_csv', type=str, default='evaluation_scores.csv')

    args = parser.parse_args()
    h = load_config(args.config)

    if args.mode == 'train':
        train(args, h)
    elif args.mode == 'inference':
        assert args.checkpoint_path, "Checkpoint path must be provided for inference"
        assert args.input_path, "Input audio path must be provided for inference"
        inference_and_evaluate(args, h, mode='inference')
    elif args.mode == 'evaluate':
        assert args.checkpoint_path, "Checkpoint path must be provided for evaluation"
        assert args.test_noisy_dir and args.test_clean_dir, "Test directories must be provided"
        inference_and_evaluate(args, h, mode='evaluate')