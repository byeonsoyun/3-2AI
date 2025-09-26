import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import sounddevice as sd
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="./audio_dataset", help="Path to audio dataset")
parser.add_argument("--play", action="store_true", help="Play audio")
args = parser.parse_args()

DATASET_PATH = args.data_path
FILES = ["095522039.wav", "095522040.wav", "095522041.wav", "095522042.wav"]
SAVE_DIR = "featurextract_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
def plot_feature(feature, title, file_name):
    plt.figure(figsize=(10,4))
    plt.imshow(feature.squeeze().numpy(), origin="lower", aspect="auto", cmap="viridis")
    plt.title(title)
    plt.colorbar()
    plt.xlabel("Time")
    plt.ylabel("Feature bin")
    plt.savefig(os.path.join(SAVE_DIR, file_name))
    plt.show()

def play_audio(waveform, sample_rate):
    audio_np = waveform.numpy()
    if audio_np.ndim == 2 and audio_np.shape[0] > 1:
        audio_np = audio_np.mean(axis=0)
    else:
        audio_np = audio_np.squeeze()
    sd.play(audio_np, samplerate=sample_rate)
    sd.wait()

# -----------------------------
if __name__ == "__main__":
    for file_name in FILES:
        path = os.path.join(DATASET_PATH, file_name)
        waveform, sample_rate = torchaudio.load(path)

        # MFCC
        mfcc = torchaudio.transforms.MFCC(sample_rate=sample_rate, n_mfcc=13)(waveform)
        plot_feature(mfcc, title=f"MFCC of {file_name}",
                     file_name=f"{file_name}_mfcc.png")

        # MelSpectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
        plot_feature(mel_spec, title=f"MelSpectrogram of {file_name}",
                     file_name=f"{file_name}_melspec.png")

        if args.play:
            print(f"{file_name} 재생 중...")
            play_audio(waveform, sample_rate)
