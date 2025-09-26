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
SAVE_DIR = "sampling_results"
os.makedirs(SAVE_DIR, exist_ok=True)

# -----------------------------
def resample_audio(waveform, orig_sr, target_sr):
    resampler = torchaudio.transforms.Resample(orig_freq=orig_sr, new_freq=target_sr)
    return resampler(waveform)

def plot_waveform_compare(original, downsampled, upsampled, file_name):
    plt.figure(figsize=(12, 6))
    plt.subplot(3,1,1)
    plt.plot(original.t().numpy())
    plt.title(f"Original {file_name}")
    plt.subplot(3,1,2)
    plt.plot(downsampled.t().numpy())
    plt.title(f"Downsampled {file_name}")
    plt.subplot(3,1,3)
    plt.plot(upsampled.t().numpy())
    plt.title(f"Upsampled {file_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, f"{file_name}_resample_compare.png"))
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
    down_sr = 16000
    up_sr = 48000

    for file_name in FILES:
        path = os.path.join(DATASET_PATH, file_name)
        waveform, sample_rate = torchaudio.load(path)

        downsampled = resample_audio(waveform, sample_rate, down_sr)
        upsampled = resample_audio(waveform, sample_rate, up_sr)

        print(f"\n=== {file_name} ===")
        print(f"원본 SR: {sample_rate}, 다운샘플 SR: {down_sr}, 업샘플 SR: {up_sr}")

        # waveform 비교 시각화 + 저장
        plot_waveform_compare(waveform, downsampled, upsampled, file_name)

        if args.play:
            print(f"{file_name} 재생 중...")
            play_audio(waveform, sample_rate)
