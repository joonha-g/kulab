import warnings
warnings.filterwarnings("ignore")

import numpy as np
import librosa
import scipy.signal
from scipy.ndimage import median_filter
from scipy.spatial.distance import cdist
from python_speech_features import mfcc
import crepe
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


# ======================================================
# 1) 오디오 로드
# ======================================================

def load_audio(path, target_sr=16000):
    """
    오디오 파일 로드 (mono, target_sr로 리샘플).
    mp3 / wav 모두 librosa로 바로 로드합니다.
    """
    audio, sr = librosa.load(path, sr=target_sr, mono=True)
    # 정규화 (클리핑 방지)
    if np.max(np.abs(audio)) > 0:
        audio = audio / np.max(np.abs(audio))
    return audio, sr


# ======================================================
# 2) CREPE 기반 F0 추출 + 피치 시퀀스 변환
# ======================================================

def extract_f0_crepe(audio, sr, hop_length=160):
    """
    CREPE 기반 F0 추출.
    - audio: 1ch waveform
    - sr: sampling rate (16kHz 권장)
    - hop_length: 10ms @ 16kHz → 160 samples
    """
    audio_16k = audio.astype(np.float32)

    time, frequency, confidence, activation = crepe.predict(
        audio_16k,
        sr,
        viterbi=True,
        step_size=1000 * hop_length / sr  # ms 단위
    )

    f0 = np.array(frequency)
    f0[f0 <= 0] = np.nan

    # 신뢰도 낮은 구간 제거
    confidence = np.array(confidence)
    f0[confidence < 0.3] = np.nan

    # NaN 보간
    if np.all(np.isnan(f0)):
        return np.ones_like(f0) * 100.0

    nans = np.isnan(f0)
    not_nans = ~nans
    f0[nans] = np.interp(
        np.flatnonzero(nans),
        np.flatnonzero(not_nans),
        f0[not_nans]
    )

    # 약간의 median filter로 스무딩
    f0 = scipy.signal.medfilt(f0, kernel_size=3)

    return f0


def f0_to_pitch_cents(f0):
    """
    절대 F0(Hz)를 '키를 제거한 상대 피치(cents)' 시퀀스로 변환.
    - 기준: 각 곡의 중앙값 F0
    - 반환: 같은 길이의 pitch_cents 배열
    """
    f0 = np.asarray(f0, dtype=float)
    f0[f0 <= 0] = np.nan

    valid = f0[~np.isnan(f0)]
    if len(valid) == 0:
        return np.zeros_like(f0)

    median_f0 = np.median(valid)
    # 1200 * log2(f0 / median_f0) → 중앙값 기준 상대 피치 (cents)
    pitch_cents = 1200.0 * np.log2(f0 / median_f0)

    # NaN 있었던 곳은 0으로 채움 (중앙값 근처로)
    pitch_cents[np.isnan(pitch_cents)] = 0.0
    return pitch_cents


def prepare_melody_sequence(audio, sr, target_rate=20, hop_length=160):
    """
    멜로디 비교용 피치 시퀀스 생성:
      1) CREPE F0 추출
      2) 상대 피치(cents)로 변환 (키/성별 영향 감소)
      3) frame rate 다운샘플 (약 target_rate Hz)
      4) median filter로 추가 스무딩
    """
    f0 = extract_f0_crepe(audio, sr, hop_length=hop_length)
    pitch_cents = f0_to_pitch_cents(f0)

    # CREPE frame rate ≈ sr / hop_length
    orig_rate = sr / hop_length
    factor = max(1, int(round(orig_rate / target_rate)))
    pitch_ds = pitch_cents[::factor]

    # 스무딩
    if len(pitch_ds) >= 5:
        pitch_ds = median_filter(pitch_ds, size=5)

    return pitch_ds.astype(np.float32)


# ======================================================
# 3) 멜로디 유사도: DTW + 인터벌 히스토그램 + n-gram
# ======================================================

def dtw_distance(seq1, seq2):
    """
    간단한 DTW 거리 (L1 cost), O(NM).
    시퀀스 길이가 수천 수준이라면 충분히 사용 가능.
    """
    n, m = len(seq1), len(seq2)
    if n == 0 or m == 0:
        return np.inf

    seq1 = np.asarray(seq1, dtype=float)
    seq2 = np.asarray(seq2, dtype=float)

    dp = np.full((n + 1, m + 1), np.inf, dtype=float)
    dp[0, 0] = 0.0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(seq1[i - 1] - seq2[j - 1])
            dp[i, j] = cost + min(
                dp[i - 1, j],      # 삭제
                dp[i, j - 1],      # 삽입
                dp[i - 1, j - 1]   # 매칭
            )
    return dp[n, m] / (n + m)  # 평균 비용 형태로 정규화


def interval_histogram_similarity(pitch_a, pitch_b, max_interval_semitones=12):
    """
    두 피치 시퀀스에서 '음정 이동(인터벌) 분포'를 비교.
    1) pitch_cents → 반음 단위로 양자화
    2) diff → 인터벌(반음)
    3) [-max, +max] 범위에서 히스토그램 생성 후 L1 거리 → 유사도로 변환
    """
    # 반음 단위로 양자화
    midi_a = np.round(pitch_a / 100.0)
    midi_b = np.round(pitch_b / 100.0)

    if len(midi_a) < 2 or len(midi_b) < 2:
        return 0.0

    interval_a = np.diff(midi_a)
    interval_b = np.diff(midi_b)

    # 클리핑
    interval_a = np.clip(interval_a, -max_interval_semitones, max_interval_semitones)
    interval_b = np.clip(interval_b, -max_interval_semitones, max_interval_semitones)

    bins = np.arange(-max_interval_semitones - 0.5,
                     max_interval_semitones + 1.5, 1.0)

    hist_a, _ = np.histogram(interval_a, bins=bins, density=True)
    hist_b, _ = np.histogram(interval_b, bins=bins, density=True)

    # L1 거리 → 유사도 (0~100)
    l1 = 0.5 * np.sum(np.abs(hist_a - hist_b))  # [0,1] 범위 근처
    sim = max(0.0, 100.0 * (1.0 - l1))
    return sim


def motif_ngram_similarity(pitch_a, pitch_b, n=4):
    """
    피치 시퀀스에서 인터벌 기반 n-gram 모티프 유사도.
    1) pitch_cents → 반음 양자화
    2) diff → 인터벌 시퀀스
    3) n-gram 집합을 만든 뒤 Jaccard 유사도
    """
    midi_a = np.round(pitch_a / 100.0)
    midi_b = np.round(pitch_b / 100.0)

    if len(midi_a) < n + 1 or len(midi_b) < n + 1:
        return 0.0

    interval_a = np.diff(midi_a).astype(int)
    interval_b = np.diff(midi_b).astype(int)

    def build_ngrams(seq, n):
        return {tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)}

    grams_a = build_ngrams(interval_a, n)
    grams_b = build_ngrams(interval_b, n)

    if not grams_a and not grams_b:
        return 100.0
    union = grams_a | grams_b
    inter = grams_a & grams_b
    if len(union) == 0:
        return 0.0
    return 100.0 * len(inter) / len(union)


def melody_similarity_from_sequences(pitch_a, pitch_b):
    """
    멜로디 전체 유사도 계산:
      - DTW 기반 피치 윤곽 유사도
      - 인터벌 분포 유사도
      - 모티프 n-gram 유사도
      → 가중 합으로 종합 멜로디 유사도 산출
    """
    if len(pitch_a) == 0 or len(pitch_b) == 0:
        return {
            "melody_overall_similarity": 0.0,
            "pitch_dtw_similarity": 0.0,
            "interval_hist_similarity": 0.0,
            "motif_ngram_similarity": 0.0,
        }

    # 1) DTW 기반 피치 윤곽
    dtw_avg = dtw_distance(pitch_a, pitch_b)  # 단위: cents 정도의 평균 차이
    # 평균 0 cents → 100점, 600 cents(6반음) 이상이면 0 근처
    pitch_dtw_sim = max(0.0, 100.0 * (1.0 - dtw_avg / 600.0))

    # 2) 인터벌 히스토그램
    interval_sim = interval_histogram_similarity(pitch_a, pitch_b)

    # 3) 모티프 n-gram (n=4)
    ngram_sim = motif_ngram_similarity(pitch_a, pitch_b, n=4)

    # 가중 합 (멜로디 윤곽 비중을 가장 크게)
    w_dtw = 0.5
    w_hist = 0.25
    w_ngram = 0.25

    overall = w_dtw * pitch_dtw_sim + w_hist * interval_sim + w_ngram * ngram_sim

    return {
        "melody_overall_similarity": float(overall),
        "pitch_dtw_similarity": float(pitch_dtw_sim),
        "interval_hist_similarity": float(interval_sim),
        "motif_ngram_similarity": float(ngram_sim),
    }


# ======================================================
# 4) 음색/스펙트럼 특징 및 유사도
# ======================================================

def extract_timbre_features(audio, sr):
    """
    곡 전체의 음색/스펙트럼 특징 벡터 추출:
      - MFCC 평균 (20차원)
      - 스펙트럴 centroid / bandwidth / flatness 평균
    """
    # MFCC
    mfcc_feat = mfcc(audio, samplerate=sr, numcep=20)
    mfcc_mean = np.mean(mfcc_feat, axis=0)  # (20,)

    # 스펙트럴 특징
    S = np.abs(librosa.stft(audio, n_fft=2048, hop_length=512))
    centroid = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    bandwidth = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    flatness = librosa.feature.spectral_flatness(S=S).mean()

    # 로그 스케일로 안정화
    centroid_log = np.log1p(centroid)
    bandwidth_log = np.log1p(bandwidth)

    # 하나의 벡터로 합침
    feat_vec = np.concatenate([
        mfcc_mean,
        np.array([centroid_log, bandwidth_log, flatness], dtype=float)
    ])

    return feat_vec


def timbre_similarity_from_features(feat_a, feat_b):
    """
    음색 특징 벡터 두 개에서:
      - 코사인 유사도 (주요)
      - L2 거리 기반 보정
      → 0~100점 음색 유사도로 변환
    """
    feat_a = np.asarray(feat_a, dtype=float)
    feat_b = np.asarray(feat_b, dtype=float)

    # 코사인 유사도
    cos = 1.0 - cdist([feat_a], [feat_b], metric="cosine")[0][0]  # [-1,1]
    cos_norm = max(0.0, (cos + 1.0) / 2.0)  # [0,1]로 변환

    # L2 거리 기반 penalty
    l2 = np.linalg.norm(feat_a - feat_b)
    # 대략 0~50 정도 범위를 가정하고 감쇠
    penalty = np.exp(-l2 / 20.0)

    sim = 100.0 * cos_norm * penalty
    return float(sim)


# ======================================================
# 5) 리듬 특징 및 유사도 (onset envelope + tempo)
# ======================================================

def extract_rhythm_sequence(audio, sr, target_len=600):
    """
    리듬 비교용 onset envelope 시퀀스 추출:
      1) onset_strength 계산
      2) 길이를 target_len 이하로 다운샘플
      3) 평균 0, 분산 1로 정규화
    """
    onset_env = librosa.onset.onset_strength(y=audio, sr=sr)
    if len(onset_env) == 0:
        return np.zeros(1, dtype=float)

    # 길이 줄이기
    factor = max(1, int(np.ceil(len(onset_env) / target_len)))
    onset_ds = onset_env[::factor]

    # 정규화
    mu = np.mean(onset_ds)
    sigma = np.std(onset_ds)
    if sigma > 0:
        onset_ds = (onset_ds - mu) / sigma
    else:
        onset_ds = onset_ds * 0.0

    return onset_ds.astype(np.float32)


def rhythm_similarity_from_sequences(rhy_a, rhy_b, audio_a, sr_a, audio_b, sr_b):
    """
    리듬 유사도:
      - onset envelope DTW 기반 유사도
      - 추정 템포(tempo) 유사도
      → 가중 합
    """
    # 1) onset envelope DTW
    if len(rhy_a) == 0 or len(rhy_b) == 0:
        dtw_sim = 0.0
    else:
        dtw_avg = dtw_distance(rhy_a, rhy_b)
        # 평균 0 → 100, 평균 2 이상 → 거의 0
        dtw_sim = max(0.0, 100.0 * np.exp(-dtw_avg / 1.0))

    # 2) 템포 유사도
    onset_a = librosa.onset.onset_strength(y=audio_a, sr=sr_a)
    onset_b = librosa.onset.onset_strength(y=audio_b, sr=sr_b)

    if len(onset_a) == 0 or len(onset_b) == 0:
        tempo_sim = 0.0
    else:
        tempo_a = float(librosa.beat.tempo(onset_envelope=onset_a, sr=sr_a, aggregate=np.mean)[0])
        tempo_b = float(librosa.beat.tempo(onset_envelope=onset_b, sr=sr_b, aggregate=np.mean)[0])

        if tempo_a <= 0 or tempo_b <= 0:
            tempo_sim = 0.0
        else:
            diff_ratio = abs(tempo_a - tempo_b) / max(tempo_a, tempo_b)
            tempo_sim = max(0.0, 100.0 * (1.0 - diff_ratio))  # 템포가 같으면 100

    # 가중 합
    w_dtw = 0.7
    w_tempo = 0.3
    overall = w_dtw * dtw_sim + w_tempo * tempo_sim

    return {
        "rhythm_overall_similarity": float(overall),
        "rhythm_onset_dtw_similarity": float(dtw_sim),
        "rhythm_tempo_similarity": float(tempo_sim),
    }


# ======================================================
# 6) 전체 비교 파이프라인
# ======================================================

def compare_two_songs_general(path_a, path_b):
    """
    서로 다른 두 곡(또는 어떤 두 곡이든)의
    - 멜로디
    - 음색
    - 리듬
    유사도를 0~100점으로 계산하고, 표시용 dict를 반환합니다.
    """
    # 1) 오디오 로드
    audio_a, sr_a = load_audio(path_a, target_sr=16000)
    audio_b, sr_b = load_audio(path_b, target_sr=16000)

    # 2) 멜로디 시퀀스 준비
    pitch_a = prepare_melody_sequence(audio_a, sr_a)
    pitch_b = prepare_melody_sequence(audio_b, sr_b)
    melody_report = melody_similarity_from_sequences(pitch_a, pitch_b)

    # 3) 음색 특징 및 유사도
    timbre_feat_a = extract_timbre_features(audio_a, sr_a)
    timbre_feat_b = extract_timbre_features(audio_b, sr_b)
    timbre_sim = timbre_similarity_from_features(timbre_feat_a, timbre_feat_b)

    # 4) 리듬 시퀀스 및 유사도
    rhy_a = extract_rhythm_sequence(audio_a, sr_a)
    rhy_b = extract_rhythm_sequence(audio_b, sr_b)
    rhythm_report = rhythm_similarity_from_sequences(
        rhy_a, rhy_b, audio_a, sr_a, audio_b, sr_b
    )

    # 5) 전체 종합 점수
    # 멜로디 50%, 음색 25%, 리듬 25%
    overall_similarity = (
        0.5 * melody_report["melody_overall_similarity"]
        + 0.25 * timbre_sim
        + 0.25 * rhythm_report["rhythm_overall_similarity"]
    )

    report = {
        "melody": melody_report,
        "timbre_similarity": float(timbre_sim),
        "rhythm": rhythm_report,
        "overall_similarity": float(overall_similarity),
    }

    return report


# ======================================================
# 7) 자연어 설명 출력용 헬퍼
# ======================================================

def describe_similarity(report, title_a="곡 A", title_b="곡 B"):
    """
    숫자 결과를 바탕으로 간단한 한국어 설명을 출력합니다.
    """
    print("=" * 70)
    print(f"[{title_a}] vs [{title_b}] 유사도 분석 결과")
    print("=" * 70)

    overall = report["overall_similarity"]
    mel = report["melody"]["melody_overall_similarity"]
    tim = report["timbre_similarity"]
    rhy = report["rhythm"]["rhythm_overall_similarity"]

    print(f"\n[총합 유사도] : {overall:.1f} / 100")

    # 구간별 정성 평가
    def level(x):
        if x >= 80:
            return "매우 높음"
        elif x >= 60:
            return "높음"
        elif x >= 40:
            return "중간"
        else:
            return "낮음"

    print("\n[세부 점수]")
    print(f"  - 멜로디 유사도        : {mel:.1f} ({level(mel)})")
    print(f"    · 피치 윤곽(DTW)      : {report['melody']['pitch_dtw_similarity']:.1f}")
    print(f"    · 인터벌 분포         : {report['melody']['interval_hist_similarity']:.1f}")
    print(f"    · 모티프 n-gram       : {report['melody']['motif_ngram_similarity']:.1f}")
    print(f"  - 음색/사운드 질감 유사도 : {tim:.1f} ({level(tim)})")
    print(f"  - 리듬 유사도           : {rhy:.1f} ({level(rhy)})")
    print(f"    · onset 패턴(DTW)     : {report['rhythm']['rhythm_onset_dtw_similarity']:.1f}")
    print(f"    · 템포(tempo)         : {report['rhythm']['rhythm_tempo_similarity']:.1f}")

    print("\n[해석]")
    if overall >= 80:
        print("  → 두 곡은 전반적인 분위기, 멜로디 진행, 리듬이 매우 비슷한 편입니다.")
        print("    비슷한 장르/무드의 곡이거나, 강하게 영향을 받은 곡일 가능성이 큽니다.")
    elif overall >= 60:
        print("  → 여러 면에서 닮은 점이 있는 편입니다.")
        print("    같은 장르 안에서 '느낌이 비슷한 곡' 정도로 볼 수 있습니다.")
    elif overall >= 40:
        print("  → 일부 구간이나 특징만 비슷하고, 전체적으로는 다른 분위기의 곡입니다.")
    else:
        print("  → 멜로디/음색/리듬 모두 크게 다르며, 서로 다른 종류의 음악으로 보는 것이 자연스럽습니다.")

    print("\n(※ 이 평가는 '같은 곡인지 여부'를 판별하는 것이 아니라,")
    print("   서로 다른 곡들이 얼마나 닮았는지를 정량적으로 보는 용도입니다.)")
    print()

def plot_waveforms_to_pdf(path_a, path_b, pdf_path, title_prefix=""):
    """
    두 오디오의 파형을 PDF로 저장합니다.
    구성:
      1) A 파형
      2) B 파형
      3) A/B 오버랩

    기존 커버곡 비교 코드의 파형 출력과 완전히 동일합니다.
    """

    print(f"[Waveform] 파형 PDF 생성 시작 → {pdf_path}")

    # 오디오 로드
    audio_a, sr_a = librosa.load(path_a, sr=16000, mono=True)
    audio_b, sr_b = librosa.load(path_b, sr=16000, mono=True)

    # 시간축 생성
    t_a = np.linspace(0, len(audio_a) / sr_a, len(audio_a))
    t_b = np.linspace(0, len(audio_b) / sr_b, len(audio_b))

    # 오버랩을 위한 최소 길이 맞춤
    min_len = min(len(audio_a), len(audio_b))
    audio_a_ov = audio_a[:min_len]
    audio_b_ov = audio_b[:min_len]
    t_ov = np.linspace(0, min_len / sr_a, min_len)

    # PDF 생성
    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        # 1) A 파형
        axes[0].plot(t_a, audio_a)
        axes[0].set_title(f"{title_prefix} - 곡 A 파형")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Amplitude")

        # 2) B 파형
        axes[1].plot(t_b, audio_b)
        axes[1].set_title(f"{title_prefix} - 곡 B 파형")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Amplitude")

        # 3) A/B 오버랩
        axes[2].plot(t_ov, audio_a_ov, alpha=0.5, color='blue', label="A")
        axes[2].plot(t_ov, audio_b_ov, alpha=0.5, color='orange', label="B")
        axes[2].set_title(f"{title_prefix} - A/B 오버랩")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[Waveform] 파형 PDF 생성 완료: {pdf_path}")

# ======================================================
# 8) 실행 예시
# ======================================================

if __name__ == "__main__":
    # 여기만 주인님 환경에 맞게 수정하시면 됩니다.
    fileA = "Iris_out.mp3"   # 예: 비교할 곡 1
    fileB = "Lean.mp3"   # 예: 비교할 곡 2

    report = compare_two_songs_general(fileA, fileB)
    describe_similarity(report, title_a=fileA, title_b=fileB)


    plot_waveforms_to_pdf(fileA, fileB,
                      "waveform_compare.pdf",
                      title_prefix="노래 비교")