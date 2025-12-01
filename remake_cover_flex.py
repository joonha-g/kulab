import pandas as pd
import numpy as np
import librosa
import warnings
from scipy.signal import correlate

warnings.filterwarnings("ignore", category=UserWarning)

def normalize_audio(vec):
    max_val = np.max(np.abs(vec))
    if max_val > 0:
        return vec / max_val
    else:
        return vec

def trim_silence(vec, threshold=0.01):
    non_silence = np.where(np.abs(vec) > threshold)[0]
    if len(non_silence) == 0:
        return vec
    return vec[non_silence[0]:non_silence[-1] + 1]

def detect_key_chroma(y, sr):
    chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
    return np.argmax(np.mean(chroma, axis=1))

def get_chroma_segment(y1, y2, sr, i, seg_len):
    seg1 = y1[i * seg_len:(i + 1) * seg_len]
    seg2 = y2[i * seg_len:(i + 1) * seg_len]
    if len(seg1) < 1024 or len(seg2) < 1024:
        return None, None, None, None
    c1 = librosa.feature.chroma_cens(y=seg1, sr=sr)
    c2 = librosa.feature.chroma_cens(y=seg2, sr=sr)
    return seg1, seg2, c1, c2

def compute_similarity_over_time(y1, y2, sr, window_sec=1):
    seg_len = int(sr * window_sec)
    total_segments = min(len(y1), len(y2)) // seg_len
    sims = []
    for i in range(total_segments):
        seg1, seg2, c1, c2 = get_chroma_segment(y1, y2, sr, i, seg_len)
        if seg1 is None:
            sims.append(0)
            continue
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
        return y1[lag:], y2[:len(y1) - lag]
    else:
        return y1[:len(y2) + lag], y2[-lag:]

def hybrid_segmental_similarity(y1, y2, sr, segment_sec=1, th_low=0.6):
    seg_len = int(sr * segment_sec)
    total_segments = min(len(y1), len(y2)) // seg_len
    sims = []
    for i in range(total_segments):
        seg1, seg2, c1, c2 = get_chroma_segment(y1, y2, sr, i, seg_len)
        if seg1 is None:
            continue
        diff1 = np.mean(np.abs(c1 - c2))
        sim1 = 1.0 - diff1 / (np.max([np.std(c1), np.std(c2), 1e-6]) + 1e-6)
        if sim1 < th_low:
            raw_diff = np.mean(np.abs(seg1 - seg2))
            raw_sim = np.exp(-raw_diff * 10)
            sim1 = (sim1 * 0.7) + (raw_sim * 0.3)
        sims.append(sim1)
    return np.array(sims)

def recompare_low_segments(y1, y2, sr, sim_vector, segment_sec=2, threshold=0.6):
    seg_len = int(sr * segment_sec)
    improved = []
    for i, sim in enumerate(sim_vector):
        if sim < threshold:
            seg1, seg2, c1, c2 = get_chroma_segment(y1, y2, sr, i, seg_len)
            if seg1 is None:
                continue
            min_len = min(c1.shape[1], c2.shape[1])
            c1 = c1[:, :min_len]
            c2 = c2[:, :min_len]
            corr = np.corrcoef(c1.flatten(), c2.flatten())[0, 1]
            corr = np.clip(corr, 0, 1)
            improved.append(corr)
    return np.array(improved)

file1 = r"Iris_out_1_vector.txt"
file2 = r"Iris_out_cover_vector.txt"

vec1 = pd.read_csv(file1, sep='\t', header=None).to_numpy()
vec2 = pd.read_csv(file2, sep='\t', header=None).to_numpy()

vec1_mono = trim_silence(normalize_audio(np.mean(vec1, axis=1)))
vec2_mono = trim_silence(normalize_audio(np.mean(vec2, axis=1)))

sr = 48000

y1_h, y1_p = librosa.effects.hpss(vec1_mono)
y2_h, y2_p = librosa.effects.hpss(vec2_mono)

y1_mix = 0.7 * y1_h + 0.3 * y1_p
y2_mix = 0.7 * y2_h + 0.3 * y2_p

key1 = detect_key_chroma(y1_mix, sr)
key2 = detect_key_chroma(y2_mix, sr)
shift_steps = key1 - key2

y2_shifted = librosa.effects.pitch_shift(y2_mix, sr=sr, n_steps=shift_steps)

sim_vector_global = compute_similarity_over_time(y1_mix, y2_shifted, sr, window_sec=1)
start_idx = find_first_high_similarity_index(sim_vector_global, threshold=0.75)
start_sample = start_idx * sr

y1_sync = y1_mix[start_sample:]
y2_sync = y2_shifted[start_sample:]

y1_sync, y2_sync = align_by_crosscorr(y1_sync, y2_sync)

FLEX_WINDOWS = [round(0.8 + 0.1 * i, 1) for i in range(8)]

threshold = 0.7
good_segments = []

for w in FLEX_WINDOWS:
    sim_vector = hybrid_segmental_similarity(y1_sync, y2_sync, sr, segment_sec=w, th_low=0.6)
    for i, sim in enumerate(sim_vector):
        if sim >= threshold:
            a_start = (start_sample / sr) + i * w
            a_end = a_start + w
            b_start = i * w
            b_end = b_start + w
            good_segments.append((w, a_start, a_end, b_start, b_end, sim))

print("===== [결과 요약] =====")
for idx, (w, a_s, a_e, b_s, b_e, sim) in enumerate(good_segments):
    print(f"윈도우 {w:.1f}s | 구간 {idx+1}: A노래 {a_s:.1f}~{a_e:.1f}초 / B노래 {b_s:.1f}~{b_e:.1f}초 | 유사도 {sim*100:.1f}%")
