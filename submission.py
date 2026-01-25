import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import os
from torch import nn
import torch.nn.functional as F

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

class AudioEncoder:
    def __init__(self, weights_path='weights/encoder.pth'):
        self.device = torch.device("cpu")
        self.model = SimpleCNN()
        
        if os.path.exists(weights_path):
            try:
                state_dict = torch.load(weights_path, map_location=self.device)
                self.model.load_state_dict(state_dict)
            except Exception as e:
                print(f"Weights load error: {e}")
        
        self.model.eval()
        self.mel_spectrogram = T.MelSpectrogram(
            sample_rate=22050,
            n_fft=1024,
            n_mels=64
        )

    def get_embedding(self, audio_path):
        try:
            waveform, sr = torchaudio.load(audio_path)
            
            if sr != 22050:
                resampler = T.Resample(sr, 22050)
                waveform = resampler(waveform)

            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            spec = self.mel_spectrogram(waveform)
            log_spec = torch.log(spec + 1e-9).unsqueeze(0)

            with torch.no_grad():
                vector = self.model(log_spec)
                vector = F.normalize(vector, p=2, dim=1)
                return vector.numpy().flatten()
                
        except Exception:
            return np.zeros(512)

    def predict_track(self, noisy_audio_path, database):
        query_vector = self.get_embedding(noisy_audio_path)
        best_score = -100
        best_track_id = None

        for track_id, db_vector in database.items():
            score = np.dot(query_vector, db_vector)
            if score > best_score:
                best_score = score
                best_track_id = track_id
                
        return best_track_id