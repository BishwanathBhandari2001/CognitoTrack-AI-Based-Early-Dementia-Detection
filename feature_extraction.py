import librosa
import numpy as np

def extract_features(path):

    y, sr = librosa.load(path, duration=20)

    features = {}

    # =====================================================
    # 1️⃣ MFCC FEATURES (13 mean + 13 std)
    # =====================================================
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

    for i in range(13):
        features[f"mfcc_{i+1}_mean"] = np.mean(mfcc[i])
        features[f"mfcc_{i+1}_std"] = np.std(mfcc[i])

    # =====================================================
    # 2️⃣ DELTA MFCC (Speech Dynamics)
    # =====================================================
    delta_mfcc = librosa.feature.delta(mfcc)

    for i in range(13):
        features[f"delta_mfcc_{i+1}_mean"] = np.mean(delta_mfcc[i])

    # =====================================================
    # 3️⃣ PITCH FEATURES (Voice Stability)
    # =====================================================
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]

    if len(pitch_values) > 0:
        features["pitch_mean"] = np.mean(pitch_values)
        features["pitch_std"] = np.std(pitch_values)
        features["pitch_range"] = np.max(pitch_values) - np.min(pitch_values)
    else:
        features["pitch_mean"] = 0
        features["pitch_std"] = 0
        features["pitch_range"] = 0

    # =====================================================
    # 4️⃣ ENERGY FEATURES
    # =====================================================
    rms = librosa.feature.rms(y=y)[0]
    features["rms_mean"] = np.mean(rms)
    features["rms_std"] = np.std(rms)
    features["energy_variation"] = np.std(rms)

    # =====================================================
    # 5️⃣ ZERO CROSSING RATE
    # =====================================================
    zcr = librosa.feature.zero_crossing_rate(y)[0]
    features["zcr_mean"] = np.mean(zcr)
    features["zcr_std"] = np.std(zcr)

    # =====================================================
    # 6️⃣ SPECTRAL FEATURES
    # =====================================================
    features["spectral_centroid_mean"] = np.mean(
        librosa.feature.spectral_centroid(y=y, sr=sr)
    )

    features["spectral_bandwidth_mean"] = np.mean(
        librosa.feature.spectral_bandwidth(y=y, sr=sr)
    )

    features["spectral_rolloff_mean"] = np.mean(
        librosa.feature.spectral_rolloff(y=y, sr=sr)
    )

    features["spectral_contrast_mean"] = np.mean(
        librosa.feature.spectral_contrast(y=y, sr=sr)
    )

    # =====================================================
    # 7️⃣ TEMPO (Speech Speed)
    # =====================================================
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    features["tempo"] = tempo

    # =====================================================
    # 8️⃣ PAUSE & HESITATION FEATURES
    # =====================================================
    intervals = librosa.effects.split(y, top_db=25)

    silence_count = 0
    total_silence = 0
    prev_end = 0

    for start, end in intervals:
        gap = start - prev_end
        if gap > 0:
            silence_count += 1
            total_silence += gap
        prev_end = end

    total_length = len(y)

    features["pause_ratio"] = total_silence / total_length
    features["speech_ratio"] = 1 - features["pause_ratio"]
    features["average_pause_duration"] = (
        total_silence / silence_count if silence_count > 0 else 0
    )
    features["hesitation_rate"] = silence_count / (total_length / sr)
    features["silence_count"] = silence_count

    # =====================================================
    # 9️⃣ HARMONIC TO NOISE RATIO (HNR)
    # =====================================================
    harmonic = librosa.effects.harmonic(y)
    noise = y - harmonic

    features["hnr"] = np.mean(harmonic**2) / (np.mean(noise**2) + 1e-6)

    return features
