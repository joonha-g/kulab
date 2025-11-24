import os
import glob
import subprocess
import warnings

import numpy as np
import librosa
import parselmouth
from parselmouth.praat import call
from scipy.spatial.distance import cdist, euclidean
from dtw import dtw
from python_speech_features import mfcc
import pyworld as pw

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ======================================================
# 0) FFmpeg 경로 (주인님 PC 환경에 맞게 설정하십시오)
# ======================================================
ffmpeg_path = r"C:\Users\arjkh\scoop\apps\ffmpeg\current\bin\ffmpeg.exe"


# ======================================================
# 1) MP3 → WAV 변환 + Demucs로 원곡 보컬 분리
# ======================================================

def ensure_wav(input_path: str) -> str:
    """
    입력이 mp3면 같은 이름의 wav로 변환하고,
    이미 wav면 그대로 반환합니다.
    """
    base, ext = os.path.splitext(input_path)
    if ext.lower() == ".wav":
        return input_path

    wav_path = base + ".wav"
    if not os.path.exists(wav_path):
        print(f"[FFmpeg] {input_path} → {wav_path} 변환 중입니다...")
        cmd = [
            ffmpeg_path,
            "-y",
            "-i", input_path,
            wav_path
        ]
        subprocess.run(cmd, check=True)
        print("[FFmpeg] 변환이 완료되었습니다:", wav_path)
    else:
        print(f"[FFmpeg] 이미 존재하는 WAV를 사용합니다: {wav_path}")
    return wav_path


def run_demucs_two_stems(input_wav: str, out_root: str = "demucs_out"):
    """
    demucs --two-stems vocals -o out_root input_wav
    명령을 실행하여 보컬 / no_vocals 스템을 생성합니다.
    """
    print(f"[Demucs] 분리를 시작합니다: {input_wav}")
    cmd = [
        "demucs",
        "--two-stems", "vocals",
        "-o", out_root,
        input_wav
    ]
    subprocess.run(cmd, check=True)
    print(f"[Demucs] 분리가 완료되었습니다: {input_wav}")


def find_demucs_vocal(out_root: str, input_wav: str) -> str:
    """
    Demucs 출력 폴더 구조에서
    vocals.wav 경로를 찾아 반환합니다.
    예)
      out_root / mdx_extra_q / <basename> / vocals.wav
    """
    base = os.path.splitext(os.path.basename(input_wav))[0]
    vocal_pattern = os.path.join(out_root, "*", base, "vocals.wav")
    vocal_candidates = glob.glob(vocal_pattern)

    if not vocal_candidates:
        raise FileNotFoundError(f"vocals.wav를 찾을 수 없습니다. 패턴: {vocal_pattern}")

    vocal_path = vocal_candidates[0]
    print(f"[Demucs] 찾은 원곡 보컬 스템: {vocal_path}")
    return vocal_path


def extract_original_vocal(original_file: str, out_root: str = "demucs_out") -> str:
    """
    원곡 전체 음악 파일에서 Demucs를 이용해 '보컬 스템' 파일 경로를 얻습니다.
    """
    wav_path = ensure_wav(original_file)
    run_demucs_two_stems(wav_path, out_root=out_root)
    vocal_path = find_demucs_vocal(out_root, wav_path)
    return vocal_path


# ======================================================
# 2) 오디오 로드 함수
# ======================================================

def load_audio(path):
    """
    오디오 파일을 16kHz 모노로 로드합니다.
    """
    audio, sr = librosa.load(path, sr=16000, mono=True)
    return audio, sr


# ======================================================
# 3) WORLD 기반 F0 추출
# ======================================================

def extract_f0_world(audio, sr):
    """
    pyWORLD(Harvest + StoneMask)를 이용해 F0 시퀀스를 추출합니다.
    반환값은 무성 구간 보간까지 완료된 F0 배열(Hz)입니다.
    """
    audio = audio.astype(np.float64)

    f0, t = pw.harvest(
        audio,
        sr,
        f0_floor=50.0,
        f0_ceil=1100.0,
        frame_period=10.0,  # ms
    )
    f0 = pw.stonemask(audio, f0, t, sr)

    f0 = np.array(f0)
    f0[f0 <= 0] = np.nan

    # 전구간 무성인 극단적인 경우
    if np.all(np.isnan(f0)):
        f0[:] = 100.0
        return f0

    # NaN 구간을 인접 유효 값으로 선형 보간
    nans = np.isnan(f0)
    not_nans = ~nans
    f0[nans] = np.interp(
        np.flatnonzero(nans),
        np.flatnonzero(not_nans),
        f0[not_nans]
    )
    return f0


# ======================================================
# 4) DTW 기반 멜로디 유사도 계산
# ======================================================

def melody_similarity(f0_a, f0_b):
    """
    두 F0 시퀀스에 대해 DTW 기반 멜로디 유사도를 계산합니다.
    센트(cents) 단위 피치 차이 통계를 반환합니다.
    (원곡 보컬 vs 커버 보컬)
    """
    x = f0_a.reshape(-1, 1)
    y = f0_b.reshape(-1, 1)

    alignment = dtw(x, y, dist_method=euclidean)
    distance = alignment.distance
    idx_a = alignment.index1
    idx_b = alignment.index2

    eps = 1e-8
    cents = 1200 * np.log2((f0_a[idx_a] + eps) / (f0_b[idx_b] + eps))
    abs_cents = np.abs(cents)

    return {
        "dtw_distance": float(distance),
        "mean_cents_diff": float(np.mean(abs_cents)),
        "median_cents_diff": float(np.median(abs_cents)),
        "percent_within_25cents": float(np.mean(abs_cents < 25) * 100),
        "percent_within_1semitone": float(np.mean(abs_cents < 100) * 100),
        "alignment_length": int(len(cents)),
    }


# ======================================================
# 5) 음색 특징 (MFCC, Formant, HNR)
# ======================================================

def extract_mfcc_features(audio, sr):
    """
    MFCC를 계산하고, 시간축 평균을 취해
    13차원 평균 MFCC 벡터를 반환합니다.
    """
    mfcc_feat = mfcc(audio, samplerate=sr, numcep=13)
    mfcc_mean = np.mean(mfcc_feat, axis=0)
    return mfcc_mean


def formants_and_hnr(audio, sr):
    """
    Praat(Parselmouth)를 이용해 평균 F1,F2,F3와 평균 HNR을 추정합니다.
    """
    snd = parselmouth.Sound(audio, sr)
    formant_obj = call(snd, "To Formant (burg)", 0.005, 5, 5000, 0.03, 50)

    f1_list, f2_list, f3_list = [], [], []
    for t in np.linspace(0, snd.duration, 300):
        try:
            f1 = call(formant_obj, "Get value at time", 1, t, "Hertz")
            f2 = call(formant_obj, "Get value at time", 2, t, "Hertz")
            f3 = call(formant_obj, "Get value at time", 3, t, "Hertz")
            if f1 > 0:
                f1_list.append(f1)
            if f2 > 0:
                f2_list.append(f2)
            if f3 > 0:
                f3_list.append(f3)
        except Exception:
            pass

    harmonics = call(snd, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonics, "Get mean", 0, 0)

    F1 = float(np.mean(f1_list)) if len(f1_list) > 0 else np.nan
    F2 = float(np.mean(f2_list)) if len(f2_list) > 0 else np.nan
    F3 = float(np.mean(f3_list)) if len(f3_list) > 0 else np.nan
    hnr = float(hnr)

    return F1, F2, F3, hnr


def timbre_similarity(audio_a, sr_a, audio_b, sr_b):
    """
    두 보컬 오디오의 음색 유사도를 계산합니다.
      - MFCC 거리 / 코사인 유사도
      - Formant, HNR 차이
    (원곡 보컬 vs 커버 보컬)
    """
    # 길이를 가장 짧은 쪽에 맞춤
    min_len = min(len(audio_a), len(audio_b))
    audio_a = audio_a[:min_len]
    audio_b = audio_b[:min_len]

    # MFCC
    mfcc_a = extract_mfcc_features(audio_a, sr_a)
    mfcc_b = extract_mfcc_features(audio_b, sr_b)
    mfcc_dist = float(np.linalg.norm(mfcc_a - mfcc_b))
    mfcc_cos = 1 - float(cdist([mfcc_a], [mfcc_b], metric="cosine")[0][0])

    # Formant + HNR
    F1a, F2a, F3a, HNR_a = formants_and_hnr(audio_a, sr_a)
    F1b, F2b, F3b, HNR_b = formants_and_hnr(audio_b, sr_b)

    result = {
        "mfcc_distance": mfcc_dist,
        "mfcc_cosine_similarity": mfcc_cos,
        "formant_F1_diff": float(abs(F1a - F1b)) if not (np.isnan(F1a) or np.isnan(F1b)) else np.nan,
        "formant_F2_diff": float(abs(F2a - F2b)) if not (np.isnan(F2a) or np.isnan(F2b)) else np.nan,
        "formant_F3_diff": float(abs(F3a - F3b)) if not (np.isnan(F3a) or np.isnan(F3b)) else np.nan,
        "hnr_diff": float(abs(HNR_a - HNR_b)),
    }

    return result


# ======================================================
# 6) 점수 계산: 피치 점수 / 음색 점수 / 종합 점수
# ======================================================

def compute_pitch_score(melody_report):
    """
    멜로디 비교 결과를 바탕으로 0~100 사이의 피치 점수 계산.
    (노래방의 음정 점수 느낌)
    """
    mean_cents = melody_report.get("mean_cents_diff", 999.0)
    p25 = melody_report.get("percent_within_25cents", 0.0)

    # 평균 피치 오차가 클수록 점수 감소 (2센트당 1점씩 감소)
    pitch_from_mean = max(0.0, 100.0 - (mean_cents / 2.0))

    # 25센트 이내 구간 비율을 그대로 점수로 사용 (0~100)
    pitch_from_p25 = max(0.0, min(100.0, p25))

    pitch_score = 0.5 * pitch_from_mean + 0.5 * pitch_from_p25
    return float(max(0.0, min(100.0, pitch_score)))


def describe_pitch_score(pitch_score, melody_report):
    """
    피치 점수와 세부 지표를 바탕으로 '노래방 스타일' 설명 문장 생성.
    """
    mean_cents = melody_report.get("mean_cents_diff", 999.0)
    p25 = melody_report.get("percent_within_25cents", 0.0)
    p100 = melody_report.get("percent_within_1semitone", 0.0)

    if pitch_score >= 95:
        level = "최상급"
        comment = "원곡과 거의 동일한 수준으로 음정이 매우 정확합니다."
    elif pitch_score >= 85:
        level = "매우 높음"
        comment = "전체적으로 음정이 잘 맞으며, 일부 구간에서만 작은 차이가 있습니다."
    elif pitch_score >= 70:
        level = "보통 이상"
        comment = "중요한 구간은 대체로 맞지만, 여러 구간에서 음정 차이가 느껴질 수 있습니다."
    elif pitch_score >= 50:
        level = "보통 이하"
        comment = "원곡과 다른 음정으로 부른 부분이 많아 전반적인 피치 일치도가 낮습니다."
    else:
        level = "낮음"
        comment = "멜로디 패턴이나 음정이 원곡과 상당히 다르게 부른 편입니다."

    detail = (
        f"평균 피치 오차는 약 {mean_cents:.1f} 센트이며, "
        f"25센트 이내로 맞은 프레임 비율은 {p25:.1f}%, "
        f"1반음(100센트) 이내는 {p100:.1f}%입니다."
    )

    return level, comment, detail


def compute_timbre_score(timbre_report):
    """
    음색 비교 결과를 바탕으로 0~100 사이의 음색 점수를 계산.
    (원곡 가수와의 음색 유사도 점수)
    """
    mfcc_cos = timbre_report.get("mfcc_cosine_similarity", 0.0)
    hnr_diff = timbre_report.get("hnr_diff", np.nan)

    # 코사인 유사도 (0~1) → 0~100점
    mfcc_score = max(0.0, min(1.0, mfcc_cos)) * 100.0

    # HNR 차이가 클수록 감점 (1 dB당 3점, 최대 30점 감점)
    if np.isnan(hnr_diff):
        hnr_penalty = 0.0
    else:
        hnr_penalty = min(30.0, max(0.0, hnr_diff * 3.0))

    timbre_score = mfcc_score - hnr_penalty
    return float(max(0.0, min(100.0, timbre_score)))


def describe_timbre_score(timbre_score, timbre_report):
    """
    음색 점수와 세부 지표를 바탕으로 노래방-style 설명 문장 생성.
    (원곡 가수와 얼마나 비슷한지)
    """
    mfcc_cos = timbre_report.get("mfcc_cosine_similarity", 0.0)
    mfcc_dist = timbre_report.get("mfcc_distance", 0.0)
    hnr_diff = timbre_report.get("hnr_diff", np.nan)

    if timbre_score >= 95:
        level = "거의 동일"
        comment = "원곡 가수와 매우 비슷한 음색과 발성으로 노래하고 있습니다."
    elif timbre_score >= 85:
        level = "매우 비슷함"
        comment = "음색이 상당히 비슷하여 같은 가수의 다른 테이크처럼 들릴 수 있습니다."
    elif timbre_score >= 70:
        level = "비슷한 편"
        comment = "전체적인 톤은 비슷하지만, 공명감이나 성대 사용에서 차이가 느껴집니다."
    elif timbre_score >= 50:
        level = "다소 차이 있음"
        comment = "음색 특징이 원곡과 꽤 달라 다른 가수처럼 들릴 수 있습니다."
    else:
        level = "상당히 다름"
        comment = "발성 방식과 음색 특성이 원곡과 많이 달라 별개의 보컬 스타일로 들립니다."

    if np.isnan(hnr_diff):
        detail = (
            f"MFCC 코사인 유사도는 {mfcc_cos:.3f}, "
            f"MFCC 거리(L2)는 {mfcc_dist:.2f}입니다. "
            f"HNR 차이는 계산되지 않았습니다."
        )
    else:
        detail = (
            f"MFCC 코사인 유사도는 {mfcc_cos:.3f}, "
            f"MFCC 거리(L2)는 {mfcc_dist:.2f}, "
            f"HNR 차이는 약 {hnr_diff:.2f} dB입니다."
        )

    return level, comment, detail


# ======================================================
# 7) 파형 시각화 + PDF 저장
# ======================================================

def plot_waveforms_to_pdf(path_a, path_b, pdf_path, title_prefix="보컬 비교"):
    """
    두 보컬 오디오의 파형:
      1) A(원곡 보컬)
      2) B(커버 보컬)
      3) A/B 오버랩
    을 한 PDF 파일로 저장합니다.
    """
    print(f"[Waveform] 파형 PDF 생성을 시작합니다 → {pdf_path}")
    audio_a, sr_a = load_audio(path_a)
    audio_b, sr_b = load_audio(path_b)

    # 시간축
    t_a = np.linspace(0, len(audio_a) / sr_a, num=len(audio_a))
    t_b = np.linspace(0, len(audio_b) / sr_b, num=len(audio_b))

    # 오버랩용: 길이 맞추기
    min_len = min(len(audio_a), len(audio_b))
    audio_a_ov = audio_a[:min_len]
    audio_b_ov = audio_b[:min_len]
    t_ov = np.linspace(0, min_len / sr_a, num=min_len)

    with PdfPages(pdf_path) as pdf:
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=False)

        # 1) 원곡 보컬 파형
        axes[0].plot(t_a, audio_a)
        axes[0].set_title(f"{title_prefix} - 원곡 보컬 파형")
        axes[0].set_xlabel("Time [s]")
        axes[0].set_ylabel("Amplitude")

        # 2) 커버 보컬 파형
        axes[1].plot(t_b, audio_b)
        axes[1].set_title(f"{title_prefix} - 커버 보컬 파형")
        axes[1].set_xlabel("Time [s]")
        axes[1].set_ylabel("Amplitude")

        # 3) 오버랩
        axes[2].plot(t_ov, audio_a_ov, alpha=0.5, label="원곡 보컬")
        axes[2].plot(t_ov, audio_b_ov, alpha=0.5, label="커버 보컬")
        axes[2].set_title(f"{title_prefix} - 보컬 파형 오버랩")
        axes[2].set_xlabel("Time [s]")
        axes[2].set_ylabel("Amplitude")
        axes[2].legend(loc="upper right")

        plt.tight_layout()
        pdf.savefig(fig)
        plt.close(fig)

    print(f"[Waveform] 파형 PDF 생성이 완료되었습니다: {pdf_path}")


# ======================================================
# 8) 전체 파이프라인: 원곡 보컬 vs 입력 보컬
# ======================================================

def run_vocal_similarity(original_file, cover_vocal_file, out_root="demucs_out"):
    """
    원곡 전체 음악 파일과 '보컬만 들어있는 파일'을 입력받아:
      1) 원곡에서 보컬 스템을 추출하고
      2) 두 보컬의 피치/음색을 비교하여
      3) 노래방 스타일의 점수와 해설을 출력하고
      4) 보컬 파형 PDF를 생성합니다.
    """
    print("========================================")
    print(" [1] 원곡에서 보컬 스템 추출 (Demucs)")
    print("========================================")
    original_vocal_path = extract_original_vocal(original_file, out_root=out_root)

    print("\n========================================")
    print(" [2] 커버 보컬 파일 준비 (WAV 변환 포함)")
    print("========================================")
    cover_vocal_wav = ensure_wav(cover_vocal_file)

    print("\n========================================")
    print(" [3] 보컬 오디오 로드")
    print("========================================")
    audio_ori, sr_ori = load_audio(original_vocal_path)
    audio_cov, sr_cov = load_audio(cover_vocal_wav)

    print("\n========================================")
    print(" [4] 피치(F0) 기반 멜로디 비교")
    print("========================================")
    f0_ori = extract_f0_world(audio_ori, sr_ori)
    f0_cov = extract_f0_world(audio_cov, sr_cov)
    melody_report = melody_similarity(f0_ori, f0_cov)

    print("\n========================================")
    print(" [5] 음색(MFCC + Formant + HNR) 비교")
    print("========================================")
    timbre_report = timbre_similarity(audio_ori, sr_ori, audio_cov, sr_cov)

    print("\n========================================")
    print(" [6] 피치 점수 / 음색 점수 / 종합 점수 계산")
    print("========================================")
    pitch_score = compute_pitch_score(melody_report)
    timbre_score = compute_timbre_score(timbre_report)

    # 노래방 스타일 종합 점수 (피치 70%, 음색 30%)
    overall_score = 0.7 * pitch_score + 0.3 * timbre_score

    pitch_level, pitch_comment, pitch_detail = describe_pitch_score(pitch_score, melody_report)
    timbre_level, timbre_comment, timbre_detail = describe_timbre_score(timbre_score, timbre_report)

    print("\n========== 보컬 비교 결과 (노래방 스타일) ==========")
    print(f"[종합 점수] {overall_score:.1f} / 100점")
    print("  (피치 70% + 음색 30% 기준입니다.)")

    print(f"\n[피치 점수] {pitch_score:.1f} / 100점 ({pitch_level})")
    print(f" - 설명: {pitch_comment}")
    print(f" - 세부 지표: {pitch_detail}")

    print(f"\n[음색 점수] {timbre_score:.1f} / 100점 ({timbre_level})")
    print(f" - 설명: {timbre_comment}")
    print(f" - 세부 지표: {timbre_detail}")
    print("===================================================")

    print("\n========================================")
    print(" [7] 보컬 파형 PDF 생성")
    print("========================================")
    plot_waveforms_to_pdf(
        original_vocal_path,
        cover_vocal_wav,
        pdf_path="vocal_waveforms_compare.pdf",
        title_prefix="보컬 비교"
    )

    print("\n[완료] 보컬 유사도 분석이 완료되었습니다.")
    print(" - 원곡에서 분리한 보컬 파일:", original_vocal_path)
    print(" - 커버 보컬(분석용 WAV):", cover_vocal_wav)
    print(" - 파형 PDF: vocal_waveforms_compare.pdf")


# ======================================================
# 9) 실행 예시
# ======================================================

if __name__ == "__main__":
    # 아래 두 경로만 주인님 환경에 맞게 수정하시면 됩니다.
    original_file = "Iris_out.mp3"          # 원곡 전체 음악
    cover_vocal_file = "vocals.wav"  # 보컬만 있는 커버 파일(녹음된 파일)

    run_vocal_similarity(original_file, cover_vocal_file)
