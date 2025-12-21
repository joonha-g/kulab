import numpy as np
import librosa
import warnings
from scipy.signal import correlate

warnings.filterwarnings("ignore", category=UserWarning)

def load_and_preprocess(path, target_sr=48000):
    try:
        audio, _ = librosa.load(path, sr=target_sr, mono=True)
    except Exception as e:
        return None

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        audio = audio / max_val

    threshold = 0.01
    non_silence = np.where(np.abs(audio) > threshold)[0]
    if len(non_silence) == 0:
        return audio
    return audio[non_silence[0]:non_silence[-1] + 1]


def detect_key_chroma(audio, sr):
    chroma = librosa.feature.chroma_cqt(y=audio, sr=sr)
    return np.argmax(np.mean(chroma, axis=1))


def align_by_crosscorr_with_offset(audio_a, audio_b):
    corr = correlate(audio_a, audio_b, mode='full', method='fft')
    lag = np.argmax(corr) - len(audio_b)

    if lag > 0:
        return audio_a[lag:], audio_b[:len(audio_a) - lag], lag, 0
    else:
        return audio_a[:len(audio_b) + lag], audio_b[-lag:], 0, -lag


def compute_similarity_optimized(audio_a, audio_b, sr, segment_sec=1, th_low=0.6):
    chroma_full_a = librosa.feature.chroma_cens(y=audio_a, sr=sr, hop_length=512)
    chroma_full_b = librosa.feature.chroma_cens(y=audio_b, sr=sr, hop_length=512)

    frames_per_sec = sr / 512
    frames_per_seg = int(segment_sec * frames_per_sec)
    sample_per_seg = int(segment_sec * sr)

    min_frames = min(chroma_full_a.shape[1], chroma_full_b.shape[1])
    total_segments = min_frames // frames_per_seg

    sims = []

    for i in range(total_segments):
        f_start = i * frames_per_seg
        f_end = (i + 1) * frames_per_seg

        chunk_a = chroma_full_a[:, f_start:f_end]
        chunk_b = chroma_full_b[:, f_start:f_end]

        diff = np.mean(np.abs(chunk_a - chunk_b))
        sim = 1.0 - diff / (np.max([np.std(chunk_a), np.std(chunk_b), 1e-6]) + 1e-6)

        if th_low > 0 and sim < th_low:
            s_start = i * sample_per_seg
            s_end = (i + 1) * sample_per_seg

            if s_end > len(audio_a) or s_end > len(audio_b):
                break

            seg_a_t = audio_a[s_start:s_end]
            seg_b_t = audio_b[s_start:s_end]

            raw_diff = np.mean(np.abs(seg_a_t - seg_b_t))
            raw_sim = np.exp(-raw_diff * 10)
            sim = (sim * 0.7) + (raw_sim * 0.3)

        sims.append(sim)

    return np.array(sims)

def run_analysis(path_a, path_b):

    try:
        SR = librosa.get_samplerate(path_a)
    except Exception as e:
        return 0, [], []

    audio_a = load_and_preprocess(path_a, target_sr=SR)
    audio_b = load_and_preprocess(path_b, target_sr=SR)

    if audio_a is None or audio_b is None:
        return 0, [], []

    harm_a, perc_a = librosa.effects.hpss(audio_a)
    harm_b, perc_b = librosa.effects.hpss(audio_b)

    mix_a = 0.7 * harm_a + 0.3 * perc_a
    mix_b = 0.7 * harm_b + 0.3 * perc_b

    key_a = detect_key_chroma(mix_a, SR)
    key_b = detect_key_chroma(mix_b, SR)
    semitone_shift = key_a - key_b

    audio_b_shifted = librosa.effects.pitch_shift(mix_b, sr=SR, n_steps=semitone_shift)

    sim_global = compute_similarity_optimized(mix_a, audio_b_shifted, SR, segment_sec=1, th_low=-1)

    start_idx = 0
    for i, sim in enumerate(sim_global):
        if sim >= 0.75:
            start_idx = i
            break

    start_sample = start_idx * SR

    audio_a_sync = mix_a[start_sample:]
    audio_b_sync = audio_b_shifted[start_sample:]

    audio_a_final, audio_b_final, offset_a, offset_b = align_by_crosscorr_with_offset(audio_a_sync, audio_b_sync)

    segment_sec = 1
    sim_profile = compute_similarity_optimized(audio_a_final, audio_b_final, SR, segment_sec=segment_sec, th_low=0.6)

    threshold = 0.7
    matches = []

    for i, sim in enumerate(sim_profile):
        if sim >= threshold:
            current_seg_samples = i * segment_sec * SR

            start_a = (start_sample + offset_a + current_seg_samples) / SR
            end_a = start_a + segment_sec

            start_b = (start_sample + offset_b + current_seg_samples) / SR
            end_b = start_b + segment_sec

            matches.append((start_a, end_a, start_b, end_b, sim))
    if matches:
        score = np.mean([m[4] for m in matches]) * 100
    else:
        score = 0

    vec1 = matches
    vec2 = []

    return score, vec1, vec2