import numpy as np
import librosa
import warnings
from scipy.spatial.distance import cdist

warnings.filterwarnings("ignore", category=UserWarning)

ANALYSIS_SR = 22050
HOP_LENGTH = 512
WINDOW_SEC = 1.2
SIMILARITY_THRESHOLD = 0.80

def load_and_preprocess(path):
    try:
        y, sr = librosa.load(path, sr=ANALYSIS_SR, mono=True)

        y_trimmed, _ = librosa.effects.trim(y, top_db=60)

        if len(y_trimmed) < sr:
            return None, None

        return y_trimmed, sr
    except Exception as e:
        print(f"[plagiarism.py] 파일 로드 실패 ({path}): {e}")
        return None, None


def extract_chroma_patches(y, sr, window_sec=WINDOW_SEC):
    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=HOP_LENGTH)

    frames_per_sec = sr / HOP_LENGTH
    win_frames = int(window_sec * frames_per_sec)

    if chroma.shape[1] < win_frames:
        return None, None, frames_per_sec

    n_frames = chroma.shape[1]
    n_patches = n_frames - win_frames + 1

    patches = []
    times = []

    for i in range(0, n_patches, 1):
        patch = chroma[:, i:i + win_frames].flatten()
        patches.append(patch)
        times.append(i / frames_per_sec)

    return np.array(patches), np.array(times), frames_per_sec

def fast_compare(patches_a, times_a, patches_b, times_b):
    if len(patches_a) == 0 or len(patches_b) == 0:
        return []

    distance_matrix = cdist(patches_a, patches_b, metric='cosine')

    similarity_matrix = 1.0 - distance_matrix

    rows, cols = np.where(similarity_matrix >= SIMILARITY_THRESHOLD)

    matches = []
    for r, c in zip(rows, cols):
        sim = similarity_matrix[r, c]

        matches.append({
            "start_A": times_a[r],
            "end_A": times_a[r] + WINDOW_SEC,
            "start_B": times_b[c],
            "end_B": times_b[c] + WINDOW_SEC,
            "similarity": sim
        })

    # 유사도 높은 순 정렬
    matches = sorted(matches, key=lambda x: x['similarity'], reverse=True)
    return matches


def filter_redundant_matches(matches):
    if not matches:
        return []

    final_matches = []
    covered_a = []

    for m in matches:
        is_covered = False
        center_a = (m['start_A'] + m['end_A']) / 2

        for ca in covered_a:
            if abs(center_a - ca) < 1.0:
                is_covered = True
                break

        if not is_covered:
            final_matches.append(m)
            covered_a.append(center_a)

        if len(final_matches) >= 10:
            break

    return final_matches

def run_plagiarism_check(path_a, path_b):
    print(f"[plagiarism.py] 정밀 분석 시작: {path_a} vs {path_b}")

    # 1. 오디오 로드
    y1, sr1 = load_and_preprocess(path_a)
    y2, sr2 = load_and_preprocess(path_b)

    if y1 is None or y2 is None:
        return 0, [], []

    patches_a, times_a, _ = extract_chroma_patches(y1, sr1)
    patches_b, times_b, _ = extract_chroma_patches(y2, sr2)

    if patches_a is None or patches_b is None:
        return 0, [], []

    raw_matches = fast_compare(patches_a, times_a, patches_b, times_b)

    unique_matches = filter_redundant_matches(raw_matches)

    formatted_matches = []
    for m in unique_matches:
        formatted_matches.append((
            m['start_A'],
            m['end_A'],
            m['start_B'],
            m['end_B'],
            m['similarity']
        ))

    if formatted_matches:
        score =(np.mean([m[4] for m in formatted_matches]) * 100 - 50) * 2
    else:
        score = 0

    print(f"[plagiarism.py] 분석 완료. 점수: {score:.1f}")

    return score, formatted_matches, []