"""Сборка песни «Боевой континент»: инструментал + вокал по секциям."""
import json
import os
import sys
import wave

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import engine as E  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
VOX = os.path.join(HERE, "vocal")
OUT = os.path.join(HERE, "release")

# ------------------------------------------------------------------ секции
# (kind, bars_min, [(clip, gap_before_seconds), ...])
SECTIONS = [
    ("intro", 2, []),
    ("title", 4, [("01_title", 0.0)]),
    ("verse", 8, [("v1_a", 0.45), ("v1_b", 0.45)]),
    ("pre", 4, [("pre", 0.0)]),
    ("chorus", 8, [("ch_a", 0.0), ("ch_b", 0.05), ("hey", 0.12), ("hey_b", 0.05)]),
    ("interlude", 4, []),
    ("verse", 8, [("v2_a", 0.45), ("v2_b", 0.45)]),
    ("pre", 4, [("pre", 0.0)]),
    ("chorus", 8, [("ch_a", 0.0), ("ch_b", 0.05), ("hey", 0.12), ("hey_b", 0.05)]),
    ("bridge", 4, [("bridge", 0.0), ("bridge_b", 0.2)]),
    ("chorus", 8, [("ch_a", 0.0), ("ch_b", 0.05), ("hey", 0.12), ("hey_b", 0.05)]),
    ("outro", 4, [("out_a", 0.3), ("out_b", 0.9)]),
]

TAIL = 1.6  # хвост после последнего такта


# ------------------------------------------------------------------ загрузка
def read_wav(path):
    with wave.open(path, "rb") as w:
        ch, sw, sr, n = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        raw = w.readframes(n)
    if sw == 2:
        x = np.frombuffer(raw, dtype="<i2").astype(np.float64) / 32768.0
    elif sw == 4:
        x = np.frombuffer(raw, dtype="<i4").astype(np.float64) / 2147483648.0
    else:  # 1 байт
        x = (np.frombuffer(raw, dtype=np.uint8).astype(np.float64) - 128.0) / 128.0
    if ch > 1:
        x = x.reshape(-1, ch).mean(axis=1)
    if sr != E.SR:
        x = resample_to(x, sr, E.SR)
    return x


def resample_to(x, sr_in, sr_out):
    if sr_in == sr_out or len(x) == 0:
        return x
    n_out = int(round(len(x) * sr_out / sr_in))
    idx = np.linspace(0, len(x) - 1, n_out)
    return np.interp(idx, np.arange(len(x)), x)


def trim_silence(x, thresh_ratio=0.012, pad=0.02):
    if len(x) == 0:
        return x
    pk = np.max(np.abs(x)) or 1.0
    mask = np.abs(x) > pk * thresh_ratio
    if not mask.any():
        return x
    a = max(0, np.argmax(mask) - int(pad * E.SR))
    b = min(len(x), len(x) - np.argmax(mask[::-1]) + int(pad * E.SR))
    return x[a:b]


def _stft(x, n_fft, hop):
    w = np.hanning(n_fft)
    frames = []
    for s0 in range(0, max(1, len(x) - n_fft), hop):
        frames.append(np.fft.rfft(x[s0:s0 + n_fft] * w))
    return np.array(frames), w


def time_stretch(x, ratio, n_fft=1024, hop=256):
    """Растяжение по времени (ratio > 1 — длиннее) фазовым вокодером."""
    if abs(ratio - 1.0) < 1e-3 or len(x) < n_fft * 2:
        return np.interp(np.linspace(0, len(x) - 1, max(1, int(len(x) * ratio))),
                         np.arange(len(x)), x)
    w = np.hanning(n_fft)
    pad = np.concatenate([np.zeros(n_fft), x, np.zeros(n_fft)])
    n_frames = 1 + (len(pad) - n_fft) // hop
    hop_s = max(1, int(round(hop * ratio)))
    out_len = n_frames * hop_s + n_fft
    out = np.zeros(out_len)
    norm = np.zeros(out_len)
    prev_ph = np.zeros(n_fft // 2 + 1)
    acc = np.zeros(n_fft // 2 + 1)
    omega = 2 * np.pi * hop * np.arange(n_fft // 2 + 1) / n_fft
    for i in range(n_frames):
        fr = pad[i * hop:i * hop + n_fft] * w
        S = np.fft.rfft(fr)
        ph = np.angle(S)
        mag = np.abs(S)
        d = ph - prev_ph - omega
        d -= 2 * np.pi * np.round(d / (2 * np.pi))
        acc = acc + omega + d
        prev_ph = ph
        y = np.fft.irfft(mag * np.exp(1j * acc), n_fft) * w
        s0 = i * hop_s
        out[s0:s0 + n_fft] += y
        norm[s0:s0 + n_fft] += w ** 2
    nz = norm > 1e-6
    out[nz] /= norm[nz]
    a = n_fft
    b = min(len(out), len(pad) - n_fft)
    return out[a:b]


def pitch_shift(x, semitones):
    """Сдвиг высоты без изменения длительности (вокодер + ресемплинг)."""
    if abs(semitones) < 1e-3:
        return x
    r = 2.0 ** (semitones / 12.0)
    y = time_stretch(x, r)                     # длиннее в r раз
    idx = np.linspace(0, len(y) - 1, len(x))   # возвращаем длину
    return np.interp(idx, np.arange(len(y)), y)


def highpass(x, fc=95.0):
    a = np.exp(-2 * np.pi * fc / E.SR)
    y = np.empty_like(x)
    prev_x = 0.0
    prev_y = 0.0
    for i in range(len(x)):
        prev_y = a * (prev_y + x[i] - prev_x)
        prev_x = x[i]
        y[i] = prev_y
    return y


def highpass_fast(x, fc=95.0):
    """Быстрый ФНЧ-разностный (однополюсный) фильтр через numpy-рекурсию блоками."""
    a = float(np.exp(-2 * np.pi * fc / E.SR))
    # y[n] = a*y[n-1] + a*(x[n]-x[n-1])  -> решаем аналитически как взвешенную сумму
    # используем scipy-free трюк: рекурсия через cumsum с логарифмами слишком дорого,
    # поэтому просто IIR в цикле по уникальным значениям — на 3 мин это ~8M шагов.
    d = np.diff(np.concatenate([[0.0], x]))
    y = np.empty_like(x)
    acc = 0.0
    ap = a
    for i in range(len(x)):
        acc = ap * acc + ap * d[i]
        y[i] = acc
    return y


biquad = E.biquad
peaking = E.peaking
high_shelf = E.high_shelf


def compress(x, thresh=0.28, ratio=3.2, att_ms=8.0, rel_ms=140.0):
    env = np.abs(x)
    a_att = np.exp(-1.0 / (att_ms * E.SR / 1000.0))
    a_rel = np.exp(-1.0 / (rel_ms * E.SR / 1000.0))
    e = np.empty_like(env)
    prev = 0.0
    for i in range(len(env)):
        v = env[i]
        prev = (a_att if v > prev else a_rel) * prev + (1 - (a_att if v > prev else a_rel)) * v
        e[i] = prev
    gain = np.where(e > thresh, (thresh / np.maximum(e, 1e-9)) ** (1 - 1 / ratio), 1.0)
    gain = np.convolve(gain, np.ones(64) / 64, "same")
    return x * gain


def prep_vocal(path):
    x = read_wav(path)
    x = trim_silence(x)
    x = highpass_fast(x, 105.0)
    x = biquad(x, *peaking(260, 1.1, -3.5))    # убираем бубнёж
    x = biquad(x, *peaking(2600, 1.0, 4.0))    # разборчивость
    x = biquad(x, *peaking(5200, 1.2, 2.5))    # воздух
    # эксайтер: подмешиваем тихие верхние гармоники
    exc = biquad(x, *high_shelf(3000, 12.0))
    exc = np.tanh(exc * 4.0) / 1.0 * 0.10
    x = x + exc
    x = pitch_shift(x, -1.0)
    x = compress(x, 0.20, 3.0)
    x = np.tanh(x * 1.5) / np.tanh(1.5)
    mx = np.max(np.abs(x)) or 1.0
    x = x / mx * 0.9
    f = int(0.008 * E.SR)
    if f > 1:
        x[:f] *= np.linspace(0, 1, f) ** 1.5
        x[-f:] *= np.linspace(1, 0, f) ** 1.5
    return x


# ------------------------------------------------------------------ сборка
def main(only_instrumental=False):
    os.makedirs(OUT, exist_ok=True)
    clips = {}
    if not only_instrumental:
        missing = [c for _, _, lines in SECTIONS for c, _ in lines
                   if not os.path.exists(os.path.join(VOX, c + ".wav"))]
        missing = sorted(set(missing))
        if missing:
            print("НЕТ ВОКАЛА:", ", ".join(missing))
            return 1
        for c in sorted({c.split("#")[0] for _, _, lines in SECTIONS for c, _ in lines}):
            clips[c] = prep_vocal(os.path.join(VOX, c + ".wav"))
            print(f"вокал {c}: {len(clips[c]) / E.SR:.2f} с")

    # --- длительности секций (одинаковые клипы могут стоять дважды — имена уникализируем)
    seen = {}
    resolved = []
    for kind, bars_min, lines in SECTIONS:
        ls = []
        for c, gap in lines:
            seen[c] = seen.get(c, 0) + 1
            ls.append((f"{c}#{seen[c]}", gap))
        resolved.append((kind, bars_min, ls))
    SECTIONS[:] = resolved
    plan = []
    for kind, bars_min, lines in SECTIONS:
        need = sum(len(clips[c.split('#')[0]]) / E.SR + gap for c, gap in lines) if lines else 0.0
        bars = max(bars_min, int(np.ceil(need / E.BAR)))
        if kind in ("verse", "chorus", "pre", "bridge", "outro", "title"):
            bars = max(2, int(np.ceil(bars / 2.0)) * 2)
        plan.append((kind, bars, lines, need))

    # растягиваем вокал, если не влезает в отведённые такты
    t = 0.0
    placements = []
    for kind, bars, lines, need in plan:
        avail = bars * E.BAR
        content = sum(len(clips[c.split('#')[0]]) / E.SR for c, _ in lines)
        gaps = sum(gap for _, gap in lines)
        speed = 1.0
        if lines and content + gaps > avail - 0.2:
            speed = min(1.18, (content + gaps) / max(0.1, avail - 0.2))
        tt = t + (0.55 * E.BAR if kind == "title" else (0.15 * E.BAR if lines else 0))
        for c, gap in lines:
            base = c.split('#')[0]
            if base not in clips:
                continue
            tt += gap * 0.35
            seg = clips[base]
            if speed > 1.0:
                n2 = int(len(seg) / speed)
                seg = np.interp(np.linspace(0, len(seg) - 1, n2), np.arange(len(seg)), seg)
            dbl = None
            if base in ("ch_a", "ch_b", "out_a"):
                dbl = pitch_shift(seg, 0.35)
                dbl = np.concatenate([np.zeros(int(0.014 * E.SR)), dbl])[:len(seg) + int(0.014 * E.SR)]
            placements.append((tt, seg, dbl))
            tt += len(seg) / E.SR
        t += avail
    total = t + TAIL
    print(f"длительность: {total:.1f} с ({total/60:.2f} мин), тактов: {sum(p[1] for p in plan)}")

    # --- инструментал
    n_total = int(total * E.SR)
    stems = {k: np.zeros(n_total, dtype=np.float32) for k in
             ("kick", "snare", "hat", "tom", "crash", "bass",
              "gtrL", "gtrR", "pad", "lead", "fx")}
    t = 0.0
    for i, (kind, bars, lines, need) in enumerate(plan):
        E.section(stems, t, bars, kind, big=(kind == "chorus" and i >= 8))
        t += bars * E.BAR

    # финальный аккорд-разряд
    E.add(stems["crash"], E.crash_sig(1.0), t - 0.02)
    E.add(stems["kick"], E.kick_sig(1.0), t - 0.02)

    vocal = np.zeros(n_total, dtype=np.float32)
    vocal_dbl = np.zeros(n_total, dtype=np.float32)
    for tt, seg, dbl in placements:
        E.add(vocal, seg.astype(np.float32), tt)
        if dbl is not None:
            E.add(vocal_dbl, (dbl * 0.55).astype(np.float32), tt)

    # дилей на конец строк: посыл — вокал, 3/16 с задержкой, только в припевах
    echo = np.zeros(n_total, dtype=np.float32)
    d = int(E.BEAT * 0.75 * E.SR)
    E.add(echo, np.concatenate([np.zeros(d), vocal[:-d]]) * 0.22, 0)

    # «эй!» в финальном припеве — с октавным двойником, чтобы звучало как толпа
    chant = np.zeros(n_total, dtype=np.float32)
    for tt, seg, _ in placements:
        if seg is clips.get("hey"):
            E.add(chant, (seg * 0.8).astype(np.float32), tt + 0.012)
    mix = E.mixdown(stems, n_total, vocal=vocal, vocal_l=vocal_dbl, vocal_r=echo,
                    chant=chant)

    # фейд-аут хвоста
    fade = int(1.2 * E.SR)
    mix[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5

    mix_inst = E.mixdown(stems, n_total, vocal=None)
    fade = int(1.2 * E.SR)
    mix_inst[-fade:] *= np.linspace(1, 0, fade)[:, None] ** 1.5

    wav_path = os.path.join(OUT, "boevoy_kontinent.wav")
    write_wav(wav_path, mix)
    print("WAV:", wav_path)
    write_wav(os.path.join(OUT, "instrumental.wav"), mix_inst)
    write_wav(os.path.join(OUT, "vocals.wav"), np.stack([vocal, vocal], axis=1))

    def mp3_of(sig, path, br=192, q=2):
        try:
            import lameenc
            enc = lameenc.Encoder()
            enc.set_bit_rate(br)
            enc.set_in_sample_rate(E.SR)
            enc.set_channels(2)
            enc.set_quality(q)
            pcm = (np.clip(sig, -1, 1) * 32767).astype("<i2").tobytes()
            data = enc.encode(pcm) + enc.flush()
            with open(path, "wb") as f:
                f.write(data)
            return len(data)
        except Exception as e:  # noqa: BLE001
            print("mp3 пропущен:", e)
            return 0

    n = mp3_of(mix_inst, os.path.join(OUT, "instrumental.mp3"))
    print("instrumental.mp3", f"{n/1e6:.1f} МБ")

    n = mp3_of(mix, os.path.join(OUT, "boevoy_kontinent.mp3"))
    print("MP3:", os.path.join(OUT, "boevoy_kontinent.mp3"), f"{n/1e6:.1f} МБ")
    mp3_of(mix, os.path.join(OUT, "preview_96k.mp3"), 96, 5)

    with open(os.path.join(OUT, "timeline.json"), "w", encoding="utf-8") as f:
        json.dump([(k, b) for k, b, _, _ in plan], f, ensure_ascii=False, indent=1)
    return 0


def write_wav(path, mix):
    pcm = (np.clip(mix, -1, 1) * 32767).astype("<i2")
    with wave.open(path, "wb") as w:
        w.setnchannels(2)
        w.setsampwidth(2)
        w.setframerate(E.SR)
        w.writeframes(pcm.tobytes())


if __name__ == "__main__":
    sys.exit(main(only_instrumental="--inst" in sys.argv))
