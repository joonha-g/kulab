import numpy as np
import pandas as pd
import librosa
import warnings

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

def compute_chroma_similarity(seg1, seg2, sr):
    if len(seg1) < 1024 or len(seg2) < 1024:
        return None
    c1 = librosa.feature.chroma_cens(y=seg1, sr=sr)
    c2 = librosa.feature.chroma_cens(y=seg2, sr=sr)
    diff = np.mean(np.abs(c1 - c2))
    denom = np.max([np.std(c1), np.std(c2), 1e-6]) + 1e-6
    sim = 1.0 - diff / denom
    return sim

def flex_search(y1, y2, sr=48000, step_sec=0.1, windows=[0.8,1.2,1.5]):
    step_samples = int(sr * step_sec)
    len1 = len(y1)
    len2 = len(y2)
    results = []

    for w in windows:
        win_samples = int(sr * w)
        if win_samples < 1024:
            continue

        starts_A = range(0, len1 - win_samples + 1, step_samples)
        starts_B = range(0, len2 - win_samples + 1, step_samples)

        for a_start in starts_A:
            a_end = a_start + win_samples
            seg1 = y1[a_start:a_end]

            for b_start in starts_B:
                b_end = b_start + win_samples
                seg2 = y2[b_start:b_end]

                sim = compute_chroma_similarity(seg1, seg2, sr)
                if sim is None:
                    continue

                results.append({
                    "window": w,
                    "start_A": a_start / sr,
                    "end_A": a_end / sr,
                    "start_B": b_start / sr,
                    "end_B": b_end / sr,
                    "similarity": sim
                })

    results_sorted = sorted(results, key=lambda x: x["similarity"], reverse=True)
    return results_sorted

def optimized_compare(y1, y2, sr=48000, coarse_limit_sec=60):
    max_samples = sr * coarse_limit_sec
    y1 = y1[:max_samples]
    y2 = y2[:max_samples]

    coarse = flex_search(y1, y2, sr, step_sec=0.5, windows=[1.2])
    if len(coarse) == 0:
        return "none"

    top_n = max(3, int(len(coarse) * 0.05))
    coarse_best = coarse[:top_n]

    refined_targets = []
    for m in coarse_best:
        refined_targets.append(m)

    refined = []
    for m in refined_targets:
        w_list = [0.8, 1.2, 1.5]
        result = flex_search(y1, y2, sr, step_sec=0.1, windows=w_list)
        refined.extend(result)

    if len(refined) == 0:
        return "none"

    refined_sorted = sorted(refined, key=lambda x: x["similarity"], reverse=True)
    return refined_sorted

file1 = r"Over_the_Horizon_vector.txt"
file2 = r"Over_the_Horizon_vector.txt"

vec1 = pd.read_csv(file1, sep="\t", header=None).to_numpy()
vec2 = pd.read_csv(file2, sep="\t", header=None).to_numpy()

y1 = normalize_audio(np.mean(vec1, axis=1))
y2 = normalize_audio(np.mean(vec2, axis=1))

y1 = trim_silence(y1)
y2 = trim_silence(y2)

sr = 48000

matches = optimized_compare(y1, y2, sr=sr, coarse_limit_sec=60)

if matches == "none":
    print("none")
else:
    valid = [m for m in matches if m["similarity"] >= 0.70]

    if len(valid) > 0:

        for i, m in enumerate(valid):
            print(
                f"구간 {i+1}: A노래 {m['start_A']:.1f}~{m['end_A']:.1f}초 / "
                f"B노래 {m['start_B']:.1f}~{m['end_B']:.1f}초 | "
                f"윈도우 {m['window']:.1f}s | "
                f"유사도 {m['similarity']*100:.1f}%"
            )
    else:
        top1 = matches[0]
        if top1["similarity"] >= 0.50:

            print(
                f"A노래 {top1['start_A']:.1f}~{top1['end_A']:.1f}초 / "
                f"B노래 {top1['start_B']:.1f}~{top1['end_B']:.1f}초 | "
                f"윈도우 {top1['window']:.1f}s | "
                f"유사도 {top1['similarity']*100:.1f}%"
            )
        else:
            print("none")
