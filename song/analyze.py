"""Анализ вокала: грубая оценка высоты тона (автокорреляция) по фреймам."""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as E  # noqa: E402
from render import read_wav, trim_silence  # noqa: E402

VOX = os.path.join(os.path.dirname(os.path.abspath(__file__)), "vocal")
NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]


def pitch_track(x, fmin=70.0, fmax=600.0, win=0.040, hop=0.010):
    """-> (times, midi float or nan, confidence)"""
    n_win = int(win * E.SR)
    n_hop = int(hop * E.SR)
    lo = int(E.SR / fmax)
    hi = int(E.SR / fmin)
    w = np.hanning(n_win)
    times, midis, conf = [], [], []
    for s in range(0, max(1, len(x) - n_win), n_hop):
        seg = x[s:s + n_win]
        if len(seg) < n_win:
            break
        seg = (seg - seg.mean()) * w
        e = float(np.dot(seg, seg))
        times.append((s + n_win / 2) / E.SR)
        if e < 1e-7:
            midis.append(np.nan)
            conf.append(0.0)
            continue
        spec = np.fft.rfft(seg, 2 * n_win)
        ac = np.fft.irfft(np.abs(spec) ** 2)[:hi + 1]
        ac /= (ac[0] + 1e-12)
        seg_ac = ac[lo:hi + 1]
        if len(seg_ac) == 0:
            midis.append(np.nan)
            conf.append(0.0)
            continue
        k = int(np.argmax(seg_ac))
        c = float(seg_ac[k])
        lag = lo + k
        # параболическая интерполяция
        if 0 < k < len(seg_ac) - 1:
            y0, y1, y2 = seg_ac[k - 1], seg_ac[k], seg_ac[k + 1]
            den = (y0 - 2 * y1 + y2)
            delta = 0.5 * (y0 - y2) / den if abs(den) > 1e-12 else 0.0
        else:
            delta = 0.0
        f = E.SR / (lag + delta)
        midis.append(69 + 12 * np.log2(f / 440.0) if c > 0.35 else np.nan)
        conf.append(c)
    return np.array(times), np.array(midis), np.array(conf)


def analyse(path):
    x = trim_silence(read_wav(path))
    t, m, c = pitch_track(x)
    good = m[~np.isnan(m)]
    if len(good) == 0:
        return None
    # округляем до полутонов и считаем гистограмму по классам
    hist = {}
    for v in good:
        pc = int(round(v)) % 12
        hist[pc] = hist.get(pc, 0.0) + 1.0
    dur = len(x) / E.SR
    return dict(dur=dur, med=float(np.median(good)), lo=float(np.percentile(good, 5)),
                hi=float(np.percentile(good, 95)), hist=hist, voiced=len(good) / max(1, len(m)))


if __name__ == "__main__":
    for f in sorted(os.listdir(VOX)):
        if not f.endswith(".wav"):
            continue
        r = analyse(os.path.join(VOX, f))
        if r is None:
            print(f, "нет тона")
            continue
        top = sorted(r["hist"].items(), key=lambda kv: -kv[1])[:4]
        print(f"{f[:-4]:8s} {r['dur']:5.2f}с  медиана {r['med']:6.2f} "
              f"({NAMES[int(round(r['med'])) % 12]}{int(round(r['med']))//12 - 1})  "
              f"диапазон {r['lo']:.1f}-{r['hi']:.1f}  тон {r['voiced']*100:.0f}%  "
              f"ноты: " + ", ".join(f"{NAMES[p]}{round(w)}" for p, w in top))
