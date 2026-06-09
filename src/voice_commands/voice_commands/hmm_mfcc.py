#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import csv
import pickle
import warnings
import numpy as np
import scipy.io.wavfile as wav
from scipy.signal import lfilter
from scipy.fftpack import dct

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")


WORDS = [
    "start", "stop", "home", "drop",
    "left", "right", "forward", "back",
    "pick", "turn"
]

FS = 16_000
FRAME_SIZE = 320
HOP_SIZE = 128
N_MFCC = 13

CODEBOOK_SIZE = 256
N_STATES = 5

WORD_STATES = {
    "start": 6,
    "stop": 4,
    "home": 4,
    "drop": 4,
    "left": 4,
    "right": 4,
    "forward": 7,
    "back": 4,
    "pick": 4,
    "turn": 4,
}

DATA_DIR = os.path.join("data", "diego")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

MODEL_DIR = "models_hmm"
RESULT_DIR = "results_hmm"

EPS = 1e-6
LOG_ZERO = -1e12


def list_wavs(folder: str) -> list:
    if not os.path.exists(folder):
        print(f"  [WARN] No existe carpeta: {folder}")
        return []

    wavs = [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(".wav")
    ]

    wavs.sort()
    return wavs


def check_dataset():
    print("\n[Verificando dataset]")
    print(f"  TRAIN_DIR: {TRAIN_DIR}")
    print(f"  TEST_DIR : {TEST_DIR}")

    if not os.path.exists(TRAIN_DIR):
        raise RuntimeError(f"No existe carpeta de entrenamiento: {TRAIN_DIR}")

    if not os.path.exists(TEST_DIR):
        print(f"  [WARN] No existe carpeta de prueba: {TEST_DIR}")

    total_train = 0
    total_test = 0

    for word in WORDS:
        train_wavs = list_wavs(os.path.join(TRAIN_DIR, word))
        test_wavs = list_wavs(os.path.join(TEST_DIR, word))

        total_train += len(train_wavs)
        total_test += len(test_wavs)

        print(
            f"  {word:10s} | train: {len(train_wavs):3d} "
            f"| test: {len(test_wavs):3d}"
        )

    if total_train == 0:
        raise RuntimeError(
            "No se encontraron audios de entrenamiento. "
            "Revisa que existan carpetas como data/diego/train/start/*.wav"
        )

    print(f"  Total train: {total_train}")
    print(f"  Total test : {total_test}")



# AUDIO Y PREPROCESAMIENTO

def load_audio(path: str) -> np.ndarray:
    fs, data = wav.read(path)

    if fs != FS:
        raise ValueError(f"Se esperaba {FS} Hz, pero '{path}' tiene {fs} Hz")

    data = data.astype(np.float64)

    if data.ndim > 1:
        data = data.mean(axis=1)

    mx = np.max(np.abs(data))

    if mx > 0:
        data /= mx

    return data


def preemphasis(signal: np.ndarray, coeff: float = 0.95) -> np.ndarray:
    return lfilter([1.0, -coeff], [1.0], signal)


def frame_signal(
    signal: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE
) -> np.ndarray:
    window = np.hamming(frame_size)

    if len(signal) < frame_size:
        signal = np.pad(signal, (0, frame_size - len(signal)))

    n_frames = 1 + (len(signal) - frame_size) // hop_size
    frames = np.zeros((n_frames, frame_size))

    for i in range(n_frames):
        start = i * hop_size
        chunk = signal[start:start + frame_size]

        if len(chunk) < frame_size:
            chunk = np.pad(chunk, (0, frame_size - len(chunk)))

        frames[i] = chunk * window

    return frames


def detect_vad(
    signal: np.ndarray,
    frame_size: int = FRAME_SIZE,
    hop_size: int = HOP_SIZE,
    k_sigma: float = 1.2,
    margin_frames: int = 4
) -> np.ndarray:
    frames = frame_signal(signal, frame_size, hop_size)
    energy = np.array([np.sum(f ** 2) for f in frames])

    e_med = np.median(energy)
    e_std = np.std(energy)

    threshold = e_med + k_sigma * e_std
    active = energy > threshold
    idxs = np.where(active)[0]

    if len(idxs) < 2:
        peak = np.argmax(energy)
        i0 = max(0, peak - 10)
        i1 = min(len(frames) - 1, peak + 10)
    else:
        i0 = max(0, idxs[0] - margin_frames)
        i1 = min(len(frames) - 1, idxs[-1] + margin_frames)

    start_sample = i0 * hop_size
    end_sample = min(i1 * hop_size + frame_size, len(signal))

    return signal[start_sample:end_sample]


# MFCC

_MEL_FILTERS = None


def hz_to_mel(hz: float) -> float:
    return 2595 * np.log10(1 + hz / 700)


def mel_to_hz(mel: float) -> float:
    return 700 * (10 ** (mel / 2595) - 1)


def build_mel_filterbank(
    n_filters: int = 26,
    n_fft: int = 512,
    fs: int = FS
) -> np.ndarray:
    global _MEL_FILTERS

    if _MEL_FILTERS is not None:
        return _MEL_FILTERS

    mel_low = hz_to_mel(0)
    mel_high = hz_to_mel(fs / 2)

    mel_pts = np.linspace(mel_low, mel_high, n_filters + 2)
    hz_pts = mel_to_hz(mel_pts)

    bins = np.floor((n_fft + 1) * hz_pts / fs).astype(int)

    fbank = np.zeros((n_filters, n_fft // 2 + 1))

    for m in range(1, n_filters + 1):
        left = bins[m - 1]
        center = bins[m]
        right = bins[m + 1]

        for k in range(left, center):
            fbank[m - 1, k] = (k - left) / (center - left + 1e-12)

        for k in range(center, right):
            fbank[m - 1, k] = (right - k) / (right - center + 1e-12)

    _MEL_FILTERS = fbank

    return fbank


def compute_mfcc(
    frame: np.ndarray,
    n_mfcc: int = N_MFCC,
    n_fft: int = 512
) -> np.ndarray:
    spectrum = np.abs(np.fft.rfft(frame, n_fft)) ** 2
    fbank = build_mel_filterbank(n_fft=n_fft)

    mel_energy = np.dot(fbank, spectrum)
    mel_energy = np.where(mel_energy > 0, mel_energy, 1e-10)

    log_mel = np.log(mel_energy)

    mfcc = dct(log_mel, type=2, norm="ortho")[:n_mfcc]

    return mfcc


def extract_mfcc(path: str) -> np.ndarray:
    audio = load_audio(path)
    audio = preemphasis(audio)
    segment = detect_vad(audio)

    frames = frame_signal(segment)
    feats = np.array([compute_mfcc(f) for f in frames])

    if len(feats) == 0:
        feats = np.zeros((1, N_MFCC))

    mean = feats.mean(axis=0)
    std = np.maximum(feats.std(axis=0), 1e-4)

    feats = (feats - mean) / std

    return feats


# ============================================================
# VQ — CODEBOOK GLOBAL
# ============================================================

def lbg_vq(
    data: np.ndarray,
    target_size: int = CODEBOOK_SIZE,
    eps: float = 1e-3,
    max_iter: int = 100
) -> np.ndarray:
    dim = data.shape[1]
    codebook = data.mean(axis=0, keepdims=True)

    while len(codebook) < target_size:
        perturb = eps * (np.ones(dim) + 1e-4 * np.random.randn(dim))

        codebook = np.vstack([
            codebook + perturb,
            codebook - perturb
        ])

        prev_distortion = np.inf
        distortion = np.inf

        for iteration in range(max_iter):
            diff = data[:, None, :] - codebook[None, :, :]
            dists = np.sum(diff ** 2, axis=2)

            labels = np.argmin(dists, axis=1)

            new_codebook = np.zeros_like(codebook)
            empty_clusters = 0

            min_dist_each_point = np.min(dists, axis=1)

            for k in range(len(codebook)):
                members = data[labels == k]

                if len(members) > 0:
                    new_codebook[k] = members.mean(axis=0)
                else:
                    new_codebook[k] = data[np.argmax(min_dist_each_point)]
                    empty_clusters += 1

            if empty_clusters > 0:
                print(
                    f"    [WARN] {empty_clusters} cluster(s) vacío(s) "
                    f"en split {len(codebook) // 2}→{len(codebook)}"
                )

            diff_new = data[:, None, :] - new_codebook[None, :, :]
            dists_new = np.sum(diff_new ** 2, axis=2)
            min_dists = np.min(dists_new, axis=1)

            distortion = np.mean(min_dists)

            codebook = new_codebook

            relative_change = abs(prev_distortion - distortion) / (
                prev_distortion + 1e-12
            )

            if relative_change < eps:
                break

            prev_distortion = distortion

        if len(codebook) > target_size:
            codebook = codebook[:target_size]

        print(
            f"  Codebook actual: {len(codebook)} vectores "
            f"(distorsión={distortion:.6f})"
        )

    return codebook


def quantize(feats: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    diff = feats[:, None, :] - codebook[None, :, :]
    dists = np.sum(diff ** 2, axis=2)

    return np.argmin(dists, axis=1).astype(int)


def train_global_codebook() -> np.ndarray:
    all_feats = []

    print("\n[1] Extrayendo MFCC para entrenar codebook global...")

    for word in WORDS:
        word_dir = os.path.join(TRAIN_DIR, word)
        wav_files = list_wavs(word_dir)

        if len(wav_files) == 0:
            print(f"  [WARN] Sin audios para: {word_dir}")
            continue

        print(f"  {word:10s}: {len(wav_files)} audios")

        for path in wav_files:
            feats = extract_mfcc(path)
            all_feats.append(feats)

    if not all_feats:
        raise RuntimeError("No se encontraron audios de entrenamiento.")

    all_feats = np.vstack(all_feats)

    print(f"  Total de vectores MFCC para VQ: {len(all_feats)}")
    print(f"  Entrenando codebook global de {CODEBOOK_SIZE} vectores...")

    codebook = lbg_vq(all_feats, CODEBOOK_SIZE)

    os.makedirs(MODEL_DIR, exist_ok=True)

    cb_path = os.path.join(MODEL_DIR, "global_codebook_256.pkl")

    with open(cb_path, "wb") as f:
        pickle.dump(codebook, f)

    np.save("codebook.npy", codebook)

    print(f"  ✓ Codebook global guardado en {cb_path}")
    print("  ✓ Copia guardada en codebook.npy")

    return codebook


# ============================================================
# HMM BAKIS
# ============================================================

def segment_sequence(
    seq: np.ndarray,
    n_states: int = N_STATES
) -> list:
    T = len(seq)

    if T < n_states:
        seq = np.pad(seq, (0, n_states - T), mode="wrap")
        T = len(seq)

    bounds = np.linspace(0, T, n_states + 1).astype(int)

    segments = []

    for s in range(n_states):
        start = bounds[s]
        end = max(bounds[s + 1], start + 1)

        segments.append(seq[start:end])

    return segments


def train_hmm_from_counts(
    sequences: list,
    n_states: int = N_STATES,
    n_symbols: int = CODEBOOK_SIZE
) -> dict:
    B_counts = np.zeros((n_states, n_symbols), dtype=np.float64)
    duration_lists = [[] for _ in range(n_states)]

    for seq in sequences:
        segs = segment_sequence(seq, n_states)

        for s_idx, seg in enumerate(segs):
            duration_lists[s_idx].append(len(seg))

            for symbol in seg:
                B_counts[s_idx, symbol] += 1

    B = B_counts + EPS
    B = B / B.sum(axis=1, keepdims=True)

    A = np.zeros((n_states, n_states), dtype=np.float64)

    for i in range(n_states):
        if i == n_states - 1:
            A[i, i] = 1.0
        else:
            avg_dur = np.mean(duration_lists[i])

            if avg_dur <= 1.0:
                p_self = 0.0
                p_next = 1.0
            else:
                p_self = 1.0 - (1.0 / avg_dur)
                p_next = 1.0 / avg_dur

            A[i, i] = p_self
            A[i, i + 1] = p_next

    row_sums = A.sum(axis=1, keepdims=True)
    A = A / row_sums

    pi = np.zeros(n_states, dtype=np.float64)
    pi[0] = 1.0

    return {
        "A": A,
        "B": B,
        "pi": pi,
        "n_states": n_states,
        "n_symbols": n_symbols
    }


def train_all_hmms(codebook: np.ndarray) -> dict:
    models = {}

    print("\n[2] Convirtiendo audios a secuencias VQ y entrenando HMMs...")

    for word in WORDS:
        sequences = []
        word_dir = os.path.join(TRAIN_DIR, word)
        wav_files = list_wavs(word_dir)

        print(f"\n  Palabra: {word}")

        if len(wav_files) == 0:
            print(f"    [ERROR] Sin datos para '{word}' — palabra omitida")
            continue

        for path in wav_files:
            feats = extract_mfcc(path)
            seq = quantize(feats, codebook)
            sequences.append(seq)

            print(
                f"    {os.path.basename(path)} → {len(seq)} símbolos "
                f"[{seq.min()}–{seq.max()}]"
            )

        n_st = WORD_STATES.get(word, N_STATES)

        models[word] = train_hmm_from_counts(
            sequences,
            n_states=n_st,
            n_symbols=CODEBOOK_SIZE
        )

        print(f"    ✓ HMM entrenado para '{word}' ({n_st} estados)")

    os.makedirs(MODEL_DIR, exist_ok=True)

    hmm_path = os.path.join(MODEL_DIR, "hmm_models.pkl")

    with open(hmm_path, "wb") as f:
        pickle.dump(models, f)

    with open("hmms.pkl", "wb") as f:
        pickle.dump(models, f)

    print(f"\n  ✓ Modelos HMM guardados en {hmm_path}")
    print("  ✓ Copia guardada en hmms.pkl")

    return models



# FORWARD EN LOG


def safe_log(x: np.ndarray) -> np.ndarray:
    return np.where(x > 0, np.log(x), LOG_ZERO)


def logsumexp(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    finite = values[values > LOG_ZERO / 2]

    if len(finite) == 0:
        return LOG_ZERO

    max_val = np.max(finite)

    return max_val + np.log(np.sum(np.exp(finite - max_val)))


def forward_log(seq: np.ndarray, hmm: dict) -> float:
    A = hmm["A"]
    B = hmm["B"]
    pi = hmm["pi"]

    logA = safe_log(A)
    logB = safe_log(B)
    logPi = safe_log(pi)

    T = len(seq)
    N = A.shape[0]

    alpha = np.full((T, N), LOG_ZERO, dtype=np.float64)

    alpha[0, :] = logPi + logB[:, seq[0]]

    for t in range(1, T):
        sym = seq[t]

        for j in range(N):
            candidates = [alpha[t - 1, j] + logA[j, j]]

            if j > 0 and A[j - 1, j] > 0:
                candidates.append(alpha[t - 1, j - 1] + logA[j - 1, j])

            alpha[t, j] = logB[j, sym] + logsumexp(np.array(candidates))

    return logsumexp(alpha[T - 1, :])


def recognize(seq: np.ndarray, models: dict) -> tuple:
    scores = {
        word: forward_log(seq, hmm)
        for word, hmm in models.items()
    }

    best = max(scores, key=scores.get)

    return best, scores


# ============================================================
# EVALUACIÓN
# ============================================================

def evaluate(codebook: np.ndarray, models: dict) -> tuple:
    print("\n[3] Evaluando con audios de prueba...")

    label2idx = {w: i for i, w in enumerate(WORDS)}

    y_true = []
    y_pred = []
    rows = []

    for word in WORDS:
        word_dir = os.path.join(TEST_DIR, word)
        wav_files = list_wavs(word_dir)

        if len(wav_files) == 0:
            print(f"  [WARN] Sin audios de prueba para: {word_dir}")
            continue

        for path in wav_files:
            feats = extract_mfcc(path)
            seq = quantize(feats, codebook)

            pred, scores = recognize(seq, models)

            y_true.append(label2idx[word])
            y_pred.append(label2idx[pred])

            mark = "✓" if pred == word else "✗"

            print(f"  {os.path.basename(path):30s} → {pred:10s} {mark}")

            row = {
                "file": path,
                "true": word,
                "pred": pred,
                "correct": pred == word
            }

            for k, v in scores.items():
                row[f"score_{k}"] = v

            rows.append(row)

    n = len(WORDS)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1

    acc = np.trace(cm) / np.sum(cm) if np.sum(cm) > 0 else 0.0

    print(f"\n  Exactitud HMM + VQ + MFCC: {acc * 100:.1f}%")

    save_evaluation_files(cm, acc, rows)

    return cm, acc


def save_evaluation_files(cm: np.ndarray, acc: float, rows: list):
    os.makedirs(RESULT_DIR, exist_ok=True)

    cm_csv_path = os.path.join(RESULT_DIR, "confusion_matrix.csv")

    with open(cm_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["true/pred"] + WORDS)

        for i, word in enumerate(WORDS):
            writer.writerow([word] + cm[i].tolist())

    eval_txt_path = os.path.join(RESULT_DIR, "evaluation.txt")

    with open(eval_txt_path, "w", encoding="utf-8") as f:
        f.write("Evaluación HMM + VQ + MFCC\n")
        f.write("==========================\n\n")
        f.write(f"Accuracy: {acc * 100:.2f}%\n\n")
        f.write("Confusion Matrix:\n")
        f.write(str(cm))
        f.write("\n\n")
        f.write("Words:\n")
        f.write(str(WORDS))
        f.write("\n")

    pred_csv_path = os.path.join(RESULT_DIR, "predictions.csv")

    if rows:
        fieldnames = list(rows[0].keys())

        with open(pred_csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    print(f"  ✓ CSV matriz de confusión → {cm_csv_path}")
    print(f"  ✓ Evaluación TXT → {eval_txt_path}")

    if rows:
        print(f"  ✓ Predicciones CSV → {pred_csv_path}")


# ============================================================
# VERIFICACIONES
# ============================================================

def verify_probabilities(models: dict):
    print("\n[Verificación de probabilidades]")

    all_ok = True

    for word, hmm in models.items():
        A_sums = hmm["A"].sum(axis=1)
        B_sums = hmm["B"].sum(axis=1)

        a_ok = np.allclose(A_sums, 1.0, atol=1e-9)
        b_ok = np.allclose(B_sums, 1.0, atol=1e-9)

        status = "✓" if (a_ok and b_ok) else "✗"

        print(
            f"  {word:10s}  "
            f"A∑={np.round(A_sums, 6)}  "
            f"B∑={np.round(B_sums, 4)}  {status}"
        )

        if not (a_ok and b_ok):
            all_ok = False

    if all_ok:
        print("  → Todas las matrices suman exactamente 1 ✓")


def print_example_sequence(codebook: np.ndarray):
    word = WORDS[0]
    word_dir = os.path.join(TRAIN_DIR, word)
    wav_files = list_wavs(word_dir)

    if len(wav_files) == 0:
        return

    path = wav_files[0]

    feats = extract_mfcc(path)
    seq = quantize(feats, codebook)

    print("\n[Ejemplo de representación VQ]")
    print(f"  Archivo  : {path}")
    print(f"  Secuencia: {seq.tolist()}")
    print(f"  Min idx  : {seq.min()}  |  Max idx: {seq.max()}")
    print(f"  Longitud : {len(seq)} símbolos")


# ============================================================
# GRÁFICAS
# ============================================================

def plot_confusion_matrix(cm: np.ndarray, acc: float):
    os.makedirs(RESULT_DIR, exist_ok=True)

    fig, ax = plt.subplots(figsize=(11, 9))

    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax, fraction=0.046)

    ax.set_xticks(range(len(WORDS)))
    ax.set_yticks(range(len(WORDS)))

    ax.set_xticklabels(WORDS, rotation=45, ha="right", fontsize=11)
    ax.set_yticklabels(WORDS, fontsize=11)

    thresh = cm.max() / 2 if cm.max() > 0 else 0

    for i in range(len(WORDS)):
        for j in range(len(WORDS)):
            color = "white" if cm[i, j] > thresh else "black"

            ax.text(
                j,
                i,
                cm[i, j],
                ha="center",
                va="center",
                fontsize=12,
                color=color
            )

    ax.set_xlabel("Predicción")
    ax.set_ylabel("Etiqueta real")

    ax.set_title(
        f"Matriz de Confusión — HMM + VQ + MFCC\n"
        f"Exactitud: {acc * 100:.1f}%"
    )

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, "confusion_matrix_hmm_vq_mfcc.png")

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"  ✓ Matriz de confusión → {path}")


def plot_A_matrix(hmm: dict, word: str):
    A = hmm["A"]

    fig, ax = plt.subplots(figsize=(6, 5))

    im = ax.imshow(A, cmap="Blues", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax)

    ax.set_title(f"Matriz A — HMM Bakis — '{word}'")
    ax.set_xlabel("Estado siguiente")
    ax.set_ylabel("Estado actual")

    ax.set_xticks(range(A.shape[1]))
    ax.set_yticks(range(A.shape[0]))

    for i in range(A.shape[0]):
        for j in range(A.shape[1]):
            ax.text(
                j,
                i,
                f"{A[i, j]:.2f}",
                ha="center",
                va="center",
                color="white" if A[i, j] > 0.5 else "black"
            )

    plt.tight_layout()

    path = os.path.join(RESULT_DIR, f"A_matrix_{word}.png")

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"  ✓ Matriz A guardada → {path}")


def plot_B_state(hmm: dict, word: str, state: int = 0):
    probs = hmm["B"][state]

    fig, ax = plt.subplots(figsize=(12, 4))

    ax.bar(np.arange(CODEBOOK_SIZE), probs)

    threshold_line = 2.0 / CODEBOOK_SIZE

    ax.axhline(
        threshold_line,
        color="red",
        linestyle="--",
        linewidth=0.8,
        label=f"2× uniforme ({threshold_line:.4f})"
    )

    ax.legend(fontsize=9)

    ax.set_title(f"Sparsity de B — '{word}' — Estado {state + 1}")
    ax.set_xlabel("Índice del codebook 0–255")
    ax.set_ylabel("Probabilidad de emisión")

    plt.tight_layout()

    path = os.path.join(
        RESULT_DIR,
        f"B_sparsity_{word}_state_{state + 1}.png"
    )

    fig.savefig(path, dpi=150)
    plt.close(fig)

    print(f"  ✓ Sparsity B guardada → {path}")


def generate_plots(models: dict, cm: np.ndarray, acc: float):
    print("\n[4] Generando gráficas...")

    plot_confusion_matrix(cm, acc)

    example_word = "start"

    if example_word in models:
        plot_A_matrix(models[example_word], example_word)
        plot_B_state(models[example_word], example_word, state=0)


# ============================================================
# MAIN
# ============================================================

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(RESULT_DIR, exist_ok=True)

    print("\n=============================================================")
    print(" RECONOCEDOR DE PALABRAS AISLADAS — HMM + VQ + MFCC")
    print("=============================================================")
    print(f" Palabras        : {WORDS}")
    print(f" Codebook global : {CODEBOOK_SIZE} vectores")
    print(" Estados HMM     : variable por palabra")
    for w, n in WORD_STATES.items():
        print(f"   {w:10s} → {n} estados")
    print(f" MFCC            : {N_MFCC} coeficientes")
    print(f" Trama           : {FRAME_SIZE} muestras / hop {HOP_SIZE}")
    print("=============================================================")

    check_dataset()

    codebook = train_global_codebook()
    models = train_all_hmms(codebook)

    print_example_sequence(codebook)
    verify_probabilities(models)

    cm, acc = evaluate(codebook, models)

    generate_plots(models, cm, acc)

    print("\n=============================================================")
    print(" RESULTADOS GENERADOS")
    print("=============================================================")
    print(f" Modelos:")
    print(f"  • {MODEL_DIR}/global_codebook_256.pkl")
    print(f"  • {MODEL_DIR}/hmm_models.pkl")
    print(f"  • codebook.npy")
    print(f"  • hmms.pkl")
    print(f"\n Resultados:")
    print(f"  • {RESULT_DIR}/confusion_matrix.csv")
    print(f"  • {RESULT_DIR}/evaluation.txt")
    print(f"  • {RESULT_DIR}/predictions.csv")
    print(f"  • {RESULT_DIR}/confusion_matrix_hmm_vq_mfcc.png")
    print(f"  • {RESULT_DIR}/A_matrix_start.png")
    print(f"  • {RESULT_DIR}/B_sparsity_start_state_1.png")
    print("\n Listo.")


if __name__ == "__main__":
    main()