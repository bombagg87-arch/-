"""Нарезка длинных вокальных клипов по паузам между фразами."""
import os
import sys
import wave
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as E  # noqa: E402
from render import read_wav, trim_silence  # noqa: E402

VOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocal")

SPLITS = {
    # файл: (доля длительности, где резать, имена половинок)
    "v1_a": (0.50, "v1_b"),
    "v2_a": (0.50, "v2_b"),
    "pre": (0.50, "pre_b"),
    "ch_a": (0.50, "ch_b"),
    "out_a": (0.55, "out_b"),
}


def write_wav(path, x, sr=E.SR):
    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def find_cut(x, target_s):
    """Ищем середину самой длинной тишины рядом с целевой точкой."""
    win = int(0.02 * E.SR)
    hop = int(0.01 * E.SR)
    env = np.array([np.sqrt(np.mean(x[i:i + win] ** 2)) for i in range(0, max(1, len(x) - win), hop)])
    if len(env) == 0:
        return int(target_s * E.SR)
    thr = np.percentile(env, 12) * 1.6
    silent = env < max(thr, np.max(env) * 0.02)
    times = np.arange(len(env)) * hop / E.SR
    # группируем паузы
    runs = []
    i = 0
    while i < len(silent):
        if silent[i]:
            j = i
            while j < len(silent) and silent[j]:
                j += 1
            runs.append((times[i], times[j - 1], j - i))
            i = j
        else:
            i += 1
    if not runs:
        return int(target_s * E.SR)
    # лучшая пауза: длинная и близкая к цели
    score = lambda r: r[2] - 30 * abs((r[0] + r[1]) / 2 - target_s)
    best = max(runs, key=score)
    return int(((best[0] + best[1]) / 2) * E.SR)


def main():
    for name, (ratio, second) in SPLITS.items():
        path = os.path.join(VOX, name + ".wav")
        if not os.path.exists(path):
            continue
        x = trim_silence(read_wav(path))
        cut = find_cut(x, len(x) / E.SR * ratio)
        pad = int(0.05 * E.SR)
        a, b = x[:cut + pad], x[max(0, cut - pad):]
        write_wav(path, a)
        write_wav(os.path.join(VOX, second + ".wav"), b)
        print(f"{name}: {len(a)/E.SR:.2f} с + {second}: {len(b)/E.SR:.2f} с (рез на {cut/E.SR:.2f} с)")


if __name__ == "__main__":
    main()
