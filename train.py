import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
import torchaudio.transforms as T
import torch.nn.functional as F
import os
import glob
import random

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 512) 

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.gap(x)
        x = x.flatten(1)
        return self.fc(x)

class AudioAugmentations(nn.Module):
    def __init__(self):
        super().__init__()
        self.freq_mask = T.FrequencyMasking(freq_mask_param=15)
        self.time_mask = T.TimeMasking(time_mask_param=35)

    def forward(self, spec):
        spec = self.freq_mask(spec)
        spec = self.time_mask(spec)
        return spec

def train_epoch(model, audio_files, optimizer, augmenter, device):
    model.train()
    batch_size = 8
    mel_transform = T.MelSpectrogram(sample_rate=22050, n_mels=64).to(device)
    
    random.shuffle(audio_files)

    for i in range(0, len(audio_files), batch_size):
        batch_files = audio_files[i:i+batch_size]
        if len(batch_files) < 2: continue

        view_1, view_2 = [], []
        
        for f in batch_files:
            try:
                wav, sr = torchaudio.load(f)
                if sr != 22050: wav = T.Resample(sr, 22050)(wav)
                wav = torch.mean(wav, dim=0, keepdim=True)
                
                target_len = 44100
                if wav.shape[1] > target_len: wav = wav[:, :target_len]
                else: wav = F.pad(wav, (0, target_len - wav.shape[1]))
                
                wav = wav.to(device)
                spec = torch.log(mel_transform(wav) + 1e-9)

                view_1.append(augmenter(spec))
                view_2.append(augmenter(spec))
            except:
                continue
        
        if len(view_1) < 2: continue

        x_i = torch.stack(view_1)
        x_j = torch.stack(view_2)

        optimizer.zero_grad()
        z_i = F.normalize(model(x_i), dim=1)
        z_j = F.normalize(model(x_j), dim=1)

        loss = (1 - (z_i * z_j).sum(dim=1)).mean()
        loss.backward()
        optimizer.step()

if __name__ == "__main__":
    # UPDATE THIS PATH
    DATA_DIR = r"C:/Users/Ashwin R/Downloads/fma_small"
    
    if not os.path.exists("weights"): os.makedirs("weights")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    files = glob.glob(os.path.join(DATA_DIR, "**/*.mp3"), recursive=True)
    if not files:
        files = glob.glob(os.path.join(DATA_DIR, "**/*.wav"), recursive=True)
    
    model = SimpleCNN().to(device)
    augmenter = AudioAugmentations().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print(f"Training on {len(files)} files...")
    
    try:
        for epoch in range(10):
            train_epoch(model, files[:200], optimizer, augmenter, device)
            torch.save(model.state_dict(), "weights/encoder.pth")
            print(f"Epoch {epoch+1} complete. Weights saved.")
    except KeyboardInterrupt:
        torch.save(model.state_dict(), "weights/encoder.pth")