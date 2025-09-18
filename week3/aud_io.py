import torch
import torchaudio
import matplotlib.pyplot as plt
import io
import os
import tarfile
import tempfile
import boto3
import requests
from botocore import UNSIGNED
from botocore.config import Config
import simpleaudio as sa  # <- 추가: 오디오 재생용
from torchaudio.utils import download_asset

# -----------------------------
# 샘플 파일 다운로드
SAMPLE_GSM = download_asset("tutorial-assets/steam-train-whistle-daniel_simon.gsm")
SAMPLE_WAV = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav")
SAMPLE_WAV_8000 = download_asset("tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042-8000hz.wav")

# -----------------------------
# 1. 로컬 파일 정보 확인
metadata = torchaudio.info(SAMPLE_WAV)
print(metadata)

# -----------------------------
# 2. HTTP 파일 다운로드 후 BytesIO 사용
url = "https://download.pytorch.org/torchaudio/tutorial-assets/steam-train-whistle-daniel_simon.wav"
response = requests.get(url)
buffer = io.BytesIO(response.content)
metadata_http = torchaudio.info(buffer)
print(metadata_http)

# waveform 로드
waveform, sample_rate = torchaudio.load(SAMPLE_WAV)

# -----------------------------
# 3. 파형 시각화 함수
def plot_waveform(waveform, sample_rate):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape
    time_axis = torch.arange(0, num_frames) / sample_rate

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].plot(time_axis, waveform[c], linewidth=1)
        axes[c].grid(True)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle("waveform")
    plt.show()

plot_waveform(waveform, sample_rate)

# -----------------------------
# 4. 스펙트로그램 시각화 함수
def plot_specgram(waveform, sample_rate, title="Spectrogram"):
    waveform = waveform.numpy()
    num_channels, num_frames = waveform.shape

    figure, axes = plt.subplots(num_channels, 1)
    if num_channels == 1:
        axes = [axes]
    for c in range(num_channels):
        axes[c].specgram(waveform[c], Fs=sample_rate)
        if num_channels > 1:
            axes[c].set_ylabel(f"Channel {c+1}")
    figure.suptitle(title)
    plt.show()

plot_specgram(waveform, sample_rate)

# -----------------------------
# 5. Audio 재생 함수 (Windows 콘솔용)
def play_audio(waveform, sample_rate):
    waveform = waveform.numpy()
    # 다중 채널이면 모노로 변환
    if waveform.shape[0] > 1:
        waveform = waveform.mean(axis=0)
    # float32 -> 16bit PCM
    audio_data = (waveform * 32767).astype("int16")
    play_obj = sa.play_buffer(audio_data, 1, 2, sample_rate)
    play_obj.wait_done()  # 재생 종료까지 대기

# 로컬 파일 재생
play_audio(waveform, sample_rate)

# -----------------------------
# 6. HTTP 데이터 로드 및 재생
url = "https://download.pytorch.org/torchaudio/tutorial-assets/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
response = requests.get(url)
buffer = io.BytesIO(response.content)
waveform_http, sample_rate_http = torchaudio.load(buffer)
plot_specgram(waveform_http, sample_rate_http, title="HTTP datasource")
play_audio(waveform_http, sample_rate_http)

# -----------------------------
# 7. TAR 파일에서 로드 및 재생
tar_path = download_asset("tutorial-assets/VOiCES_devkit.tar.gz")
tar_item = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
with tarfile.open(tar_path, mode="r") as tarfile_:
    fileobj = tarfile_.extractfile(tar_item)
    buffer_tar = io.BytesIO(fileobj.read())
    waveform_tar, sample_rate_tar = torchaudio.load(buffer_tar)
plot_specgram(waveform_tar, sample_rate_tar, title="TAR file")
play_audio(waveform_tar, sample_rate_tar)

# -----------------------------
# 8. S3에서 로드 및 재생
bucket = "pytorch-tutorial-assets"
key = "VOiCES_devkit/source-16k/train/sp0307/Lab41-SRI-VOiCES-src-sp0307-ch127535-sg0042.wav"
client = boto3.client("s3", config=Config(signature_version=UNSIGNED))
response = client.get_object(Bucket=bucket, Key=key)
buffer_s3 = io.BytesIO(response["Body"].read())
waveform_s3, sample_rate_s3 = torchaudio.load(buffer_s3)
plot_specgram(waveform_s3, sample_rate_s3, title="From S3")
play_audio(waveform_s3, sample_rate_s3)

# -----------------------------
# 9. 파일 저장 예제
def inspect_file(path):
    print("-" * 10)
    print("Source:", path)
    print("-" * 10)
    print(f" - File size: {os.path.getsize(path)} bytes")
    print(f" - {torchaudio.info(path)}")
    print()

with tempfile.TemporaryDirectory() as tempdir:
    path = f"{tempdir}/save_example_default.wav"
    torchaudio.save(path, waveform, sample_rate)
    inspect_file(path)

with tempfile.TemporaryDirectory() as tempdir:
    path = f"{tempdir}/save_example_PCM_S16.wav"
    torchaudio.save(path, waveform, sample_rate, encoding="PCM_S", bits_per_sample=16)
    inspect_file(path)

# -----------------------------
# 10. BytesIO에 저장
buffer_ = io.BytesIO()
torchaudio.save(buffer_, waveform, sample_rate, format="wav")
buffer_.seek(0)
print(buffer_.read(16))
