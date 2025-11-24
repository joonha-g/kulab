import os
import glob
import subprocess
import warnings

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.spatial.distance import cdist
from python_speech_features import mfcc
from scipy.ndimage import median_filter
import scipy.signal

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import crepe


warnings.filterwarnings("ignore")

# ==============================
# 0) 공통 상수 / 경로
# ==============================
FFMPEG_PATH = r"C:\Users\arjkh\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"
SR = 16000
HOP_LENGTH = 160          # 10 ms @ 16kHz
TARGET_FPS = 20           # 멜로디 구조 비교용 프레임률


# ==============================
# 1) F0 추출 (CREPE)
# ==============================
def extract_f0_crepe(audio, sr=SR, hop_length=HOP_LENGTH):
    """CREPE 기반 F0 시퀀스 추출."""
    audio = audio.astype(np.float32)

    time, frequency, confidence, activation = crepe.predict(
        audio,
        sr,
        viterbi=True,
        step_size=1000 * hop_length / sr  # ms
    )

    f0 = np.array(frequency)
    f0[f0 <= 0] = np.nan
    f0[confidence < 0.3] = np.nan

    if np.all(np.isnan(f0)):
        return np.ones_like(f0) * 100.0

    nans = np.isnan(f0)
    not_nans = ~nans
    f0[nans] = np.interp(np.flatnonzero(nans),
                         np.flatnonzero(not_nans),
                         f0[not_nans])

    f0 = scipy.signal.medfilt(f0, kernel_size=3)
    return f0


# ==============================
# 2) Demucs 분리 (원곡 → 보컬/MR)
# ==============================
def ensure_wav(path: str) -> str:
    base, ext = os.path.splitext(path)
    if ext.lower() == ".wav":
        return path

    wav_path = base + ".wav"
    if not os.path.exists(wav_path):
        print(f"[FFmpeg] {path} → {wav_path} 변환 중...")
        cmd = [FFMPEG_PATH, "-y", "-i", path, wav_path]
        subprocess.run(cmd, check=True)
        print("[FFmpeg] 변환 완료:", wav_path)
    else:
        print(f"[FFmpeg] 기존 WAV 사용: {wav_path}")
    return wav_path


def run_demucs_two_stems(wav_path: str, out_root: str = "demucs_out"):
    print(f"[Demucs] 분리 시작: {wav_path}")
    cmd = ["demucs", "--two-stems", "vocals", "-o", out_root, wav_path]
    subprocess.run(cmd, check=True)
    print(f"[Demucs] 분리 완료: {wav_path}")


def find_demucs_stems(out_root: str, wav_path: str):
    base = os.path.splitext(os.path.basename(wav_path))[0]
    vocal_pattern = os.path.join(out_root, "*", base, "vocals.wav")
    mr_pattern = os.path.join(out_root, "*", base, "no_vocals.wav")

    vocal = glob.glob(vocal_pattern)
    mr = glob.glob(mr_pattern)
    if not vocal:
        raise FileNotFoundError(f"vocals.wav 없음: {vocal_pattern}")
    if not mr:
        raise FileNotFoundError(f"no_vocals.wav 없음: {mr_pattern}")

    print(f"[Demucs] 보컬: {vocal[0]}")
    print(f"[Demucs] MR:    {mr[0]}")
    return vocal[0], mr[0]


def process_one_song(mp3_path: str, out_root: str = "demucs_out"):
    wav_path = ensure_wav(mp3_path)
    run_demucs_two_stems(wav_path, out_root)
    vocal, mr = find_demucs_stems(out_root, wav_path)
    return {"original": wav_path, "vocal": vocal, "mr": mr}


# ==============================
# 3) 오디오 로드
# ==============================
def load_audio(path):
    audio, sr = librosa.load(path, sr=SR, mono=True)
    return audio, sr


# ==============================
# 4) 멜로디 구조 비교 (인터벌 + n-gram)
# ==============================
def hz_to_midi(f0):
    eps = 1e-8
    midi = 69 + 12 * np.log2((f0 + eps) / 440.0)
    return np.round(midi)


def compress_notes(midi_seq):
    out = []
    for m in midi_seq:
        if not out or abs(out[-1] - m) >= 1:
            out.append(m)
    return np.array(out, dtype=np.int16)


def to_intervals(midi_seq):
    if len(midi_seq) < 2:
        return np.array([], dtype=np.int16)
    return np.diff(midi_seq).astype(np.int16)


def edit_distance(a, b):
    n, m = len(a), len(b)
    dp = np.zeros((n + 1, m + 1), dtype=int)
    for i in range(1, n + 1):
        dp[i, 0] = i
    for j in range(1, m + 1):
        dp[0, j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i, j] = min(
                dp[i - 1, j] + 1,
                dp[i, j - 1] + 1,
                dp[i - 1, j - 1] + cost
            )
    return dp[n, m]


def build_ngrams(seq, n=3):
    if len(seq) < n:
        return set()
    return {tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)}


def melody_similarity(f0_a, f0_b, sr=SR):
    """원곡-커버 멜로디 구조 유사도 계산 (인터벌 + n-gram)."""
    # 1) F0 → MIDI
    midi_a = hz_to_midi(f0_a)
    midi_b = hz_to_midi(f0_b)

    # 2) 다운샘플링 (프레임률 통일)
    orig_fps = sr / HOP_LENGTH  # ≈ 100 Hz
    factor = max(1, int(round(orig_fps / TARGET_FPS)))
    midi_a = midi_a[::factor]
    midi_b = midi_b[::factor]

    # 3) 노이즈 제거
    if len(midi_a) >= 3:
        midi_a = median_filter(midi_a, size=3)
    if len(midi_b) >= 3:
        midi_b = median_filter(midi_b, size=3)
    if len(midi_a) >= 5:
        midi_a = scipy.signal.medfilt(midi_a, kernel_size=5)
    if len(midi_b) >= 5:
        midi_b = scipy.signal.medfilt(midi_b, kernel_size=5)

    # 4) 연속 같은 음 제거
    midi_a = compress_notes(midi_a)
    midi_b = compress_notes(midi_b)

    if len(midi_a) == 0 and len(midi_b) == 0:
        return {
            "edit_distance": 0,
            "structure_length_A": 0,
            "structure_length_B": 0,
            "melody_structure_similarity": 100.0,
        }
    if len(midi_a) == 0 or len(midi_b) == 0:
        max_len = max(len(midi_a), len(midi_b))
        return {
            "edit_distance": max_len,
            "structure_length_A": int(len(midi_a)),
            "structure_length_B": int(len(midi_b)),
            "melody_structure_similarity": 0.0,
        }

    # 5) 인터벌 시퀀스
    interval_a = to_intervals(midi_a)
    interval_b = to_intervals(midi_b)

    if len(interval_a) == 0 or len(interval_b) == 0:
        seq_a = midi_a
        seq_b = midi_b
    else:
        seq_a = interval_a
        seq_b = interval_b

    # 6) Edit distance 기반 유사도
    dist = edit_distance(seq_a, seq_b)
    max_len_edit = max(len(seq_a), len(seq_b))
    if max_len_edit == 0:
        edit_sim = 100.0
    else:
        edit_sim = max(0.0, 100.0 * (1.0 - dist / max_len_edit))

    # 7) n-gram Jaccard 유사도 (인터벌 기준)
    grams_a = build_ngrams(interval_a, n=3)
    grams_b = build_ngrams(interval_b, n=3)
    if not grams_a and not grams_b:
        ngram_sim = 100.0
    else:
        union = grams_a | grams_b
        inter = grams_a & grams_b
        ngram_sim = 100.0 * len(inter) / len(union) if union else 0.0

    # 8) 가중합
    w_edit, w_ngram = 0.6, 0.4
    final_sim = w_edit * edit_sim + w_ngram * ngram_sim

    return {
        "edit_distance": int(dist),
        "structure_length_A": int(len(midi_a)),
        "structure_length_B": int(len(midi_b)),
        "melody_structure_similarity": float(final_sim),
    }


# ==============================
# 5) 음색 특징 / 유사도
# ==============================
def extract_mfcc_features(audio, sr=SR):
    mfcc_feat = mfcc(audio, samplerate=sr, numcep=13)
    return np.mean(mfcc_feat, axis=0)


def spectral_features(audio, sr=SR):
    S = np.abs(librosa.stft(audio, n_fft=1024))
    c = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
    b = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
    f = librosa.feature.spectral_flatness(S=S).mean()
    return c, b, f


def formants_and_hnr(audio, sr=SR):
    snd = parselmouth.Sound(audio, sr)
    formant_obj = call(snd, "To Formant (burg)", 0.005, 5, 5000, 0.03, 50)

    f1_list, f2_list, f3_list = [], [], []
    for t in np.linspace(0, snd.duration, 300):
        try:
            f1 = call(formant_obj, "Get value at time", 1, t, "Hertz")
            f2 = call(formant_obj, "Get value at time", 2, t, "Hertz")
            f3 = call(formant_obj, "Get value at time", 3, t, "Hertz")
            if f1 > 0: f1_list.append(f1)
            if f2 > 0: f2_list.append(f2)
            if f3 > 0: f3_list.append(f3)
        except Exception:
            pass

    harmonics = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonics, "Get mean", 0, 0)

    F1 = float(np.mean(f1_list)) if f1_list else np.nan
    F2 = float(np.mean(f2_list)) if f2_list else np.nan
    F3 = float(np.mean(f3_list)) if f3_list else np.nan
    return F1, F2, F3, float(hnr)


def timbre_similarity(audio_a, sr_a, audio_b, sr_b, use_formants=True):
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]

    mfcc_a = extract_mfcc_features(audio_a, sr_a)
    mfcc_b = extract_mfcc_features(audio_b, sr_b)
    mfcc_dist = float(np.linalg.norm(mfcc_a - mfcc_b))
    mfcc_cos = 1 - float(cdist([mfcc_a], [mfcc_b], metric="cosine")[0][0])

    c1, b1, f1 = spectral_features(audio_a, sr_a)
    c2, b2, f2 = spectral_features(audio_b, sr_b)

    result = {
        "mfcc_distance": mfcc_dist,
        "mfcc_cosine_similarity": mfcc_cos,
        "spectral_centroid_diff": float(abs(c1 - c2)),
        "spectral_bandwidth_diff": float(abs(b1 - b2)),
        "spectral_flatness_diff": float(abs(f1 - f2)),
    }

    if use_formants:
        F1a, F2a, F3a, HNR_a = formants_and_hnr(audio_a, sr_a)
        F1b, F2b, F3b, HNR_b = formants_and_hnr(audio_b, sr_b)
        result.update({
            "formant_F1_diff": float(abs(F1a - F1b)) if not (np.isnan(F1a) or np.isnan(F1b)) else np.nan,
            "formant_F2_diff": float(abs(F2a - F2b)) if not (np.isnan(F2a) or np.isnan(F2b)) else np.nan,
            "formant_F3_diff": float(abs(F3a - F3b)) if not (np.isnan(F3a) or np.isnan(F3b)) else np.nan,
            "hnr_diff": float(abs(HNR_a - HNR_b)),
        })
    return result


# ==============================
# 6) 전체 오디오 비교 (경로 기준)
# ==============================
def compare_two_audios(path_a, path_b, use_formants=True):
    audio_a, sr_a = load_audio(path_a)
    audio_b, sr_b = load_audio(path_b)

    f0_a = extract_f0_crepe(audio_a, sr_a)
    f0_b = extract_f0_crepe(audio_b, sr_b)
    melody_report = melody_similarity(f0_a, f0_b, sr_a)

    timbre_report = timbre_similarity(audio_a, sr_a, audio_b, sr_b, use_formants)
    return melody_report, timbre_report


# ==============================
# 7) 해석용 헬퍼 (커버곡 전용 문구)
# ==============================
def classify_melody_level(melody):
    sim = melody.get("melody_structure_similarity", 0.0)
    edit_d = melody.get("edit_distance", 0)
    len_a = melody.get("structure_length_A", 1)
    len_b = melody.get("structure_length_B", 1)
    max_len = max(len_a, len_b, 1)
    diff_ratio = edit_d / max_len

    if sim >= 90 and diff_ratio <= 0.1:
        return "매우 높음"
    elif sim >= 75 and diff_ratio <= 0.25:
        return "높음"
    elif sim >= 50:
        return "중간"
    else:
        return "낮음"


def describe_melody(melody):
    level = classify_melody_level(melody)
    sim = melody.get("melody_structure_similarity", 0.0)
    edit_d = melody.get("edit_distance", 0)
    len_a = melody.get("structure_length_A", 0)
    len_b = melody.get("structure_length_B", 0)
    max_len = max(len_a, len_b, 1)
    diff_ratio = edit_d / max_len * 100.0

    if level == "매우 높음":
        text = "원곡 멜로디를 거의 그대로 따라간 커버입니다."
    elif level == "높음":
        text = "전반적으로 원곡 멜로디를 잘 유지하면서 약간의 변형이 있는 커버입니다."
    elif level == "중간":
        text = "주요 구간은 비슷하지만, 해석과 변형이 비교적 많이 들어간 커버입니다."
    else:
        text = "원곡 멜로디에서 상당히 벗어난 자유로운 편곡/재해석에 가까운 커버입니다."

    detail = (
        f"멜로디 구조 유사도는 약 {sim:.1f}점, "
        f"최소 편집 횟수는 {edit_d}회, "
        f"최대 구조 길이 대비 편집 비율은 약 {diff_ratio:.1f}%입니다. "
        f"(원곡 구조 길이: {len_a}, 커버 구조 길이: {len_b})"
    )
    return level, text, detail


def classify_timbre_level(timbre):
    sim = timbre.get("mfcc_cosine_similarity", 0.0)
    dist = timbre.get("mfcc_distance", 999.0)

    if sim >= 0.95 and dist <= 5:
        return "매우 비슷함", "음색이 거의 동일하여 같은 사람·비슷한 믹싱 조건으로 들립니다."
    elif sim >= 0.90 and dist <= 9:
        return "비슷함", "음색이 상당히 비슷하여 같은 가수의 다른 테이크처럼 느껴집니다."
    elif sim >= 0.80:
        return "보통", "일부 공통된 특징은 있지만, 커버 가수의 개성이 꽤 드러납니다."
    else:
        return "다름", "음색 차이가 커서 원곡과는 다른 보컬 캐릭터로 들립니다."


def summarize_mix(melody, timbre):
    lines = []
    _, mel_text, mel_detail = describe_melody(melody)
    _, tim_text = classify_timbre_level(timbre)

    lines.append(f"멜로디 관점(전체 믹스): {mel_text}")
    lines.append(f"세부 멜로디 지표: {mel_detail}")
    lines.append(f"음색/사운드 질감 관점: {tim_text}")
    lines.append("→ 전체적으로 원곡과 커버가 얼마나 닮았는지 종합적으로 해석한 결과입니다.")
    return lines


def summarize_vocal(melody, timbre):
    lines = []
    _, mel_text, mel_detail = describe_melody(melody)
    tim_level, tim_text = classify_timbre_level(timbre)

    lines.append(f"보컬 멜로디 관점: {mel_text}")
    lines.append(f"세부 멜로디 지표: {mel_detail}")
    lines.append(f"보컬 음색 관점: {tim_text}")

    sim = timbre.get("mfcc_cosine_similarity", 0.0)
    hnr_diff = timbre.get("hnr_diff", np.nan)
    if sim >= 0.95 and (np.isnan(hnr_diff) or hnr_diff <= 2):
        singer_msg = "원곡 가수와 거의 동일한 음색으로 재현한 커버일 가능성이 매우 높습니다."
    elif sim >= 0.90:
        singer_msg = "원곡 가수와 상당히 비슷한 톤을 유지한 커버로 보입니다."
    elif sim >= 0.85:
        singer_msg = "원곡과 어느 정도 비슷하지만, 커버 가수의 개성이 함께 드러나는 수준입니다."
    else:
        singer_msg = "원곡과는 다른 보컬 캐릭터로 해석한 커버로 보입니다."

    lines.append(f"보컬 캐릭터 해석: {singer_msg}")
    return lines


def summarize_mr(melody):
    lines = []
    _, mel_text, mel_detail = describe_melody(melody)
    lines.append(f"반주/편곡 멜로디·리듬 관점: {mel_text}")
    lines.append(f"세부 구조 지표: {mel_detail}")
    lines.append("→ 원곡과 커버의 편곡이 얼마나 비슷한 스타일인지 보는 지표입니다.")
    return lines


# ==============================
# 8) 파형 시각화 + PDF
# ==============================
def plot_waveforms_to_pdf(path_a, path_b, pdf_path, title_prefix=""):
    print(f"[Waveform] PDF 생성 → {pdf_path}")
    audio_a, sr_a = load_audio(path_a)
    audio_b, sr_b = load_audio(path_b)

    t_a = np.linspace(0, len(audio_a) / sr_a, num=len(audio_a))
    t_b = np.linspace(0, len(audio_b) / sr_b, num=len(audio_b))

    min_len = min(len(audio_a), len(audio_b))
    audio_a_ov = audio_a[:min_len]
    audio_b_ov = audio_b[:min_len]
    t_ov = np.linspace(0, min_len / sr_a, num=min_len)

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8))

        axes[0].plot(t_a, audio_a)
        axes[0].set_title(f"{title_prefix} - 곡 A (원곡 추정) 파형")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Amp")

        axes[1].plot(t_b, audio_b)
        axes[1].set_title(f"{title_prefix} - 곡 B (커버 추정) 파형")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Amp")

        axes[2].plot(t_ov, audio_a_ov, alpha=0.5, label="A")
        axes[2].plot(t_ov, audio_b_ov, alpha=0.5, label="B")
        axes[2].set_title(f"{title_prefix} - A/B 파형 오버랩")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Amp")
        axes[2].legend()

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[Waveform] PDF 완료: {pdf_path}")


# ==============================
# 9) 출력 포맷
# ==============================
def print_report(title, melody, timbre, mode="mix"):
    print("\n" + "=" * 60)
    print(f"### {title} 비교 결과 ###")
    print("=" * 60)

    print("\n[멜로디 구조 지표]")
    for k in ["melody_structure_similarity", "edit_distance",
              "structure_length_A", "structure_length_B"]:
        if k in melody:
            print(f"  {k}: {melody[k]}")

    if mode != "mr" and timbre:
        print("\n[음색/스펙트럼 지표]")
        for k, v in timbre.items():
            print(f"  {k}: {v}")

    print("\n[해석 요약]")
    if mode == "mix":
        lines = summarize_mix(melody, timbre)
    elif mode == "vocal":
        lines = summarize_vocal(melody, timbre)
    else:
        lines = summarize_mr(melody)

    for line in lines:
        print(" - " + line)


# ==============================
# 10) 전체 파이프라인 (원곡 vs 커버)
# ==============================
def run_full_pipeline(fileA_mp3, fileB_mp3, out_root="demucs_out"):
    print("========================================")
    print(" [1] Demucs로 각 곡 보컬/MR 분리 (원곡 & 커버)")
    print("========================================")

    stemsA = process_one_song(fileA_mp3, out_root=out_root)
    stemsB = process_one_song(fileB_mp3, out_root=out_root)

    print("\n========================================")
    print(" [2] 원본 MIX 레벨에서 유사도")
    print("========================================")
    melody_mix, timbre_mix = compare_two_audios(
        stemsA["original"], stemsB["original"], use_formants=False
    )

    print("\n========================================")
    print(" [3] 보컬 레벨에서 유사도")
    print("========================================")
    melody_v, timbre_v = compare_two_audios(
        stemsA["vocal"], stemsB["vocal"], use_formants=True
    )

    print("\n========================================")
    print(" [4] MR(반주) 레벨에서 유사도")
    print("========================================")
    melody_mr, _ = compare_two_audios(
        stemsA["mr"], stemsB["mr"], use_formants=False
    )

    print_report("원본 MIX", melody_mix, timbre_mix, mode="mix")
    print_report("보컬",      melody_v,   timbre_v,   mode="vocal")
    print_report("MR(반주)",  melody_mr,  {},         mode="mr")

    plot_waveforms_to_pdf(stemsA["original"], stemsB["original"],
                          "waveforms_mix.pdf", title_prefix="원본 MIX")
    plot_waveforms_to_pdf(stemsA["vocal"], stemsB["vocal"],
                          "waveforms_vocal.pdf", title_prefix="보컬")
    plot_waveforms_to_pdf(stemsA["mr"], stemsB["mr"],
                          "waveforms_mr.pdf", title_prefix="MR")


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

# ==============================
# 11) 실행 예시
# ==============================
if __name__ == "__main__":
    # A: 원곡, B: 커버곡
    fileA = "Iris_out.mp3"
    fileB = "Iris_out_cover.mp3"
    run_full_pipeline(fileA, fileB)

    plot_waveforms_to_pdf(fileA, fileB,
                      "waveform_compare.pdf",
                      title_prefix="노래 비교")
