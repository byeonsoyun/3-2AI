import torch
import torchaudio
from torchaudio.utils import download_asset
import sounddevice as sd
import soundfile as sf
import matplotlib.pyplot as plt

# 1. 환경 설정
torch.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# 2. 음성 파일 다운로드
SPEECH_FILE = r"C:\Users\ByeonSoYun\3-2AI\3-2AI\week5\HW\audio.wav"



# 3. 사전 학습된 ASR 모델 불러오기
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)



# 4. 오디오 재생 (Windows 환경)
def play_audio(file_path: str):
    """주어진 wav 파일을 재생합니다."""
    data, samplerate = sf.read(file_path)
    sd.play(data, samplerate)
    sd.wait()

play_audio(SPEECH_FILE)


# 5. Waveform 로딩 + 샘플레이트 맞추기
waveform, sample_rate = torchaudio.load(SPEECH_FILE)  # [채널, 샘플수]
waveform = waveform.to(device)

# 모델이 기대하는 샘플레이트(16kHz)로 리샘플링
if sample_rate != bundle.sample_rate:
    waveform = torchaudio.functional.resample(
        waveform, sample_rate, bundle.sample_rate
    )

print(f"Waveform shape: {waveform.shape}, Sample rate: {bundle.sample_rate}")


# 6. 모델 내부 Feature 추출 + 시각화
# Transformer 레이어별 feature를 확인 가능
with torch.inference_mode():
    features, _ = model.extract_features(waveform)

fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
for i, feats in enumerate(features):
    ax[i].imshow(feats[0].cpu(), interpolation="nearest", aspect="auto", origin="lower")
    ax[i].set_title(f"Feature from transformer layer {i+1}")
    ax[i].set_xlabel("Feature dimension")
    ax[i].set_ylabel("Frame (time-axis)")

fig.tight_layout()
plt.show()



# 7. 모델 통과 후 emission(logits) 시각화
with torch.inference_mode():
    emission, _ = model(waveform)

plt.imshow(emission[0].cpu().T, interpolation="nearest", aspect="auto", origin="lower")
plt.title("Classification result")
plt.xlabel("Frame (time-axis)")
plt.ylabel("Class")
plt.tight_layout()
plt.show()

print("Class labels:", bundle.get_labels())



# 8. CTC Greedy 디코딩: emission -> 실제 텍스트로 변환
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank   # CTC blank 토큰 인덱스

    def forward(self, emission: torch.Tensor) -> str:
        """Given a sequence emission over labels, get the best path string
        Args:
            emission (Tensor): Logit tensors. Shape `[num_seq, num_label]`.

        Returns:
            str: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # 각 시간 프레임에서 가장 높은 클래스 선택
        indices = torch.unique_consecutive(indices, dim=-1)   # 연속 중복 제거 (CTC 특성)
        indices = [i for i in indices if i != self.blank]   # blank 제거
        return "".join([self.labels[i] for i in indices])   # 최종 텍스트 변환

decoder = GreedyCTCDecoder(labels=bundle.get_labels())
transcript = decoder(emission[0])
print(transcript)

play_audio(SPEECH_FILE)