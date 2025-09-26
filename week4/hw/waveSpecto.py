import os
import torch
import torchaudio
import matplotlib.pyplot as plt
import sounddevice as sd
import argparse

# -----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", default="./audio_dataset", help="Path to audio dataset")
parser.add_argument("--play", action="store_true", help="Play audio")
args = parser.parse_args()

DATASET_PATH = args.data_path
FILES = ["095522039.wav", "095522040.wav", "095522041.wav", "095522042.wav"]
SAVE_DIR = "waveSpecto_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
def plot_waveform(waveform, title="Waveform", file=None):
    plt.figure(figsize=(10, 4))
    plt.plot(waveform.t().numpy())
    plt.title(title)
    plt.xlabel("Time (samples)")
    plt.ylabel("Amplitude")
    if file:
        plt.savefig(file)
    plt.show()

def plot_spectrogram(waveform, title="Spectrogram", file=None):
    spec = torchaudio.transforms.Spectrogram()(waveform)
    plt.figure(figsize=(10, 4))
    plt.imshow(10 * torch.log10(spec).squeeze().numpy(),
               origin="lower", aspect="auto", cmap="viridis")
    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Frequency bin")
    plt.colorbar(label="dB")
    if file:
        plt.savefig(file)
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

        print(f"\n=== {file_name} ===")
        print(f"샘플링 레이트: {sample_rate}, Waveform shape: {waveform.shape}")

        # 시각화 + 저장
        plot_waveform(waveform, title=f"Waveform of {file_name}",
                      file=os.path.join(SAVE_DIR, f"{file_name}_waveform.png"))
        plot_spectrogram(waveform, title=f"Spectrogram of {file_name}",
                         file=os.path.join(SAVE_DIR, f"{file_name}_spectrogram.png"))

        # 오디오 재생
        if args.play:
            print(f"{file_name} 재생 중...")
            play_audio(waveform, sample_rate)
