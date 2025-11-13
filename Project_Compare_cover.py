import pandas as pd
import numpy as np
import librosa
import warnings
from scipy.signal import correlate
from scipy.ndimage import median_filter
import soundfile as sf

warnings.filterwarnings("ignore", category=UserWarning)

# =============================
# 기본 전처리 함수
# =============================

def normalize_audio(vec):
    max_val = np.max(np.abs(vec))
    return vec / max_val if max_val > 0 else vec

def trim_silence(vec, threshold=0.01):
    non_silence = np.where(np.abs(vec) > threshold)[0]
    if len(non_silence) == 0:
        return vec
    return vec[non_silence[0]:non_silence[-1]+1]

def detect_key_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return np.argmax(np.mean(chroma, axis=1))

def compute_similarity_over_time(y1, y2, sr, window_sec=1):
    seg_len = int(sr * window_sec)
    total_segments = min(len(y1), len(y2)) // seg_len
    sims = []
    for i in range(total_segments):
        seg1 = y1[i*seg_len:(i+1)*seg_len]
        seg2 = y2[i*seg_len:(i+1)*seg_len]
        if len(seg1) < 1024 or len(seg2) < 1024:
            sims.append(0)
            continue
        c1 = librosa.feature.chroma_cens(y=seg1, sr=sr)
        c2 = librosa.feature.chroma_cens(y=seg2, sr=sr)
        diff = np.mean(np.abs(c1 - c2))
        sim = 1.0 - diff / (np.max([np.std(c1), np.std(c2), 1e-6]) + 1e-6)
        sims.append(sim)
    return np.array(sims)

def find_first_high_similarity_index(sim_vector, threshold=0.75):
    for i, sim in enumerate(sim_vector):
        if sim >= threshold:
            return i
    return 0

def align_by_crosscorr(y1, y2):
    corr = correlate(y1, y2, mode='full')
    lag = np.argmax(corr) - len(y2)
    if lag > 0:
        return y1[lag:], y2[:len(y1)-lag]
    else:
        return y1[:len(y2)+lag], y2[-lag:]

# =============================
# 1단계: 기본 하이브리드 비교
# =============================

def hybrid_segmental_similarity(y1, y2, sr, segment_sec=1, th_low=0.6):
    seg_len = int(sr * segment_sec)
    total_segments = min(len(y1), len(y2)) // seg_len
    sims = []
    for i in range(total_segments):
        seg1 = y1[i*seg_len:(i+1)*seg_len]
        seg2 = y2[i*seg_len:(i+1)*seg_len]
        if len(seg1) < 1024 or len(seg2) < 1024:
            continue
        c1 = librosa.feature.chroma_cens(y=seg1, sr=sr)
        c2 = librosa.feature.chroma_cens(y=seg2, sr=sr)
        diff1 = np.mean(np.abs(c1 - c2))
        sim1 = 1.0 - diff1 / (np.max([np.std(c1), np.std(c2), 1e-6]) + 1e-6)
        # 파형 기반 보정
        if sim1 < th_low:
            raw_diff = np.mean(np.abs(seg1 - seg2))
            raw_sim = np.exp(-raw_diff * 10)
            sim1 = (sim1 * 0.7) + (raw_sim * 0.3)
        sims.append(sim1)
    return np.array(sims)

# =============================
# 2단계: 상관계수 기반 보정
# =============================

def recompare_low_segments(y1, y2, sr, sim_vector, segment_sec=2, threshold=0.6):
    seg_len = int(sr * segment_sec)
    improved = []
    for i, sim in enumerate(sim_vector):
        if sim < threshold:
            seg1 = y1[i*seg_len:(i+1)*seg_len]
            seg2 = y2[i*seg_len:(i+1)*seg_len]
            if len(seg1) < 1024 or len(seg2) < 1024:
                continue
            c1 = librosa.feature.chroma_cens(y=seg1, sr=sr)
            c2 = librosa.feature.chroma_cens(y=seg2, sr=sr)
            min_len = min(c1.shape[1], c2.shape[1])
            c1, c2 = c1[:, :min_len], c2[:, :min_len]
            corr = np.corrcoef(c1.flatten(), c2.flatten())[0, 1]
            corr = np.clip(corr, 0, 1)
            improved.append(corr)
    return np.array(improved)

# =============================
# 3단계: Auto-Tune 청취용 멜로디 합성
# =============================

def autotune_melody_vector_listenable(y, sr):
    f0 = librosa.yin(y, fmin=80, fmax=1000, sr=sr)
    f0 = np.nan_to_num(f0)
    f0_smooth = median_filter(f0, size=7)
    for i in range(1, len(f0_smooth)):
        if f0_smooth[i] <= 1:
            f0_smooth[i] = f0_smooth[i-1]
    hop_length = max(1, int(len(y) / max(1, len(f0_smooth))))
    t_frames = np.arange(len(f0_smooth)) * hop_length
    t = np.arange(len(y))
    freq_interp = np.interp(t, t_frames, f0_smooth)
    phase = 2 * np.pi * np.cumsum(freq_interp) / sr
    tuned = np.sin(phase)
    tuned = median_filter(tuned, size=5)
    tuned = tuned / (np.max(np.abs(tuned)) + 1e-6)
    return tuned

# =============================
# 4단계: 피치 시퀀스 기반 (형태 유사성 중심)
# =============================

def compare_pitch_sequences(y1, y2, sr):
    """피치 시퀀스 기반 유사도 (형태 중심, 상관계수 방식)"""
    f0_1 = librosa.yin(y1, fmin=80, fmax=1000, sr=sr)
    f0_2 = librosa.yin(y2, fmin=80, fmax=1000, sr=sr)
    f0_1, f0_2 = np.nan_to_num(f0_1), np.nan_to_num(f0_2)

    # 로그 스케일 + 정규화 (상대 형태 유지)
    p1 = 12 * np.log2(np.maximum(f0_1, 1e-6) / 440.0)
    p2 = 12 * np.log2(np.maximum(f0_2, 1e-6) / 440.0)
    p1 = (p1 - np.mean(p1)) / (np.std(p1) + 1e-6)
    p2 = (p2 - np.mean(p2)) / (np.std(p2) + 1e-6)

    min_len = min(len(p1), len(p2))
    p1, p2 = p1[:min_len], p2[:min_len]

    corr = np.corrcoef(p1, p2)[0, 1]
    corr = np.clip(corr, 0, 1)
    similarity = corr ** 0.5  # 감쇠 적용
    return similarity * 100

# =============================
# 메인 실행
# =============================

file1 = r"Iris_out_1_vector.txt"
file2 = r"Iris_out_cover_vector.txt"

vec1 = pd.read_csv(file1, sep='\t', header=None).to_numpy()
vec2 = pd.read_csv(file2, sep='\t', header=None).to_numpy()
vec1_mono = trim_silence(normalize_audio(np.mean(vec1, axis=1)))
vec2_mono = trim_silence(normalize_audio(np.mean(vec2, axis=1)))
sr = 48000

# 하모닉·퍼커시브 혼합
y1_h, y1_p = librosa.effects.hpss(vec1_mono)
y2_h, y2_p = librosa.effects.hpss(vec2_mono)
y1_mix = 0.7 * y1_h + 0.3 * y1_p
y2_mix = 0.7 * y2_h + 0.3 * y2_p

# 키 정렬
key1 = detect_key_chroma(y1_mix, sr)
key2 = detect_key_chroma(y2_mix, sr)
shift_steps = key1 - key2
print(f"감지된 키: 원곡={key1}, 커버={key2}, 반음 이동={shift_steps:+}")
y2_shifted = librosa.effects.pitch_shift(y2_mix, sr=sr, n_steps=shift_steps)

# 유사도 상승 구간
sim_vector_global = compute_similarity_over_time(y1_mix, y2_shifted, sr, window_sec=1)
start_idx = find_first_high_similarity_index(sim_vector_global, threshold=0.75)
start_sample = start_idx * sr
print(f"\n최초 유사도 상승 구간: {start_idx}초 부근")

# 동기화
y1_sync = y1_mix[start_sample:]
y2_sync = y2_shifted[start_sample:]
y1_sync, y2_sync = align_by_crosscorr(y1_sync, y2_sync)

# 하이브리드 비교
segment_sec = 1
sim_vector = hybrid_segmental_similarity(y1_sync, y2_sync, sr, segment_sec=segment_sec, th_low=0.6)
avg_sim = np.mean(sim_vector)
high_sim_segments = np.sum(sim_vector >= 0.9)
similar_segments = np.sum(sim_vector >= 0.7)
total_segments = len(sim_vector)
total_time = total_segments * segment_sec
similar_time = similar_segments * segment_sec

# 판정 및 보정
if high_sim_segments >= 3:
    result = "두 음악은 같은 곡입니다."
    improved_avg = None
elif avg_sim > 0.6:
    result = "두 음악은 커버곡입니다."
    improved = recompare_low_segments(y1_sync, y2_sync, sr, sim_vector, segment_sec=2, threshold=0.6)
    improved_avg = np.mean(improved) if len(improved) > 0 else None
else:
    result = "두 음악은 다른 음악입니다."
    improved_avg = None

# Auto-Tune 청취용 파일 생성
y1_auto = autotune_melody_vector_listenable(y1_sync, sr)
y2_auto = autotune_melody_vector_listenable(y2_sync, sr)

# 피치 시퀀스 유사도 계산
pitch_sim_pct = compare_pitch_sequences(y1_auto, y2_auto, sr)

# =============================
# 출력
# =============================

avg_sim_pct = avg_sim * 100
improved_avg_pct = improved_avg * 100 if improved_avg is not None else None
similar_ratio_pct = (similar_time / total_time) * 100

print("\n===== [결과 요약] =====")
print(f"총 비교 시간: {total_time:.1f}초")
print(f"유사한 시간(유사도 ≥ 70%): {similar_time:.1f}초 ({similar_ratio_pct:.1f}%)")
print(f"평균 유사도: {avg_sim_pct:.1f}%")
if improved_avg_pct is not None:
    print(f"1차 보정(상관계수) 평균 유사도: {improved_avg_pct:.1f}%")
print(f"피치 시퀀스 기반 유사도(형태 중심): {pitch_sim_pct:.1f}%")
print(f"\n→ {result}")
