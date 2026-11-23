#!/usr/bin/env python3
"""Synthesizes the score for the short film -> build/score.wav (48 kHz stereo)."""
import math, random, struct, wave, sys, os

SR = 48000
random.seed(7)

def build(total, starts, path="build/score.wav"):
    n = int(total * SR)
    buf = [0.0] * n
    # ---- slow evolving drone layers
    layers = [(55.0, 0.20), (82.41, 0.13), (110.0, 0.10), (164.81, 0.055),
              (220.0, 0.03), (41.2, 0.16)]
    for f, a in layers:
        ph = random.random() * 6.28
        lfo_f = random.uniform(0.03, 0.09)
        det = random.uniform(-0.25, 0.25)   # slight detune for width
        for i in range(n):
            t = i / SR
            env = 0.55 + 0.45 * math.sin(2 * math.pi * lfo_f * t + ph)
            v = math.sin(2 * math.pi * (f + det) * t) + 0.25 * math.sin(2 * math.pi * (f + det) * 2 * t)
            buf[i] += a * env * v

    # ---- sub pulse (heartbeat of war drums), every 2 beats at 60 bpm
    beat = 1.0
    i = 0.0
    while i < total:
        s0 = int(i * SR)
        for k in range(int(0.45 * SR)):
            j = s0 + k
            if j >= n:
                break
            t = k / SR
            buf[j] += 0.30 * math.exp(-11 * t) * math.sin(2 * math.pi * 46 * t)
        i += beat * 2

    # ---- impact hits at each scene cut (skip the very first)
    for st in starts[1:]:
        s0 = int(st * SR)
        for k in range(int(2.2 * SR)):
            j = s0 + k
            if j >= n:
                break
            t = k / SR
            a = math.exp(-3.2 * t)
            buf[j] += 0.42 * a * math.sin(2 * math.pi * 49 * t)
            buf[j] += 0.12 * a * math.sin(2 * math.pi * 98.5 * t)

    # ---- risers before the last third of the film (noise sweep)
    for st in starts[5:]:
        length = 3.0
        s0 = int((st - length) * SR)
        lp = 0.0
        idx = 0
        for k in range(int(length * SR)):
            j = s0 + k
            if j < 0 or j >= n:
                continue
            p = k / (length * SR)
            cutoff = 0.004 + 0.09 * p * p
            lp += cutoff * (random.uniform(-1, 1) - lp)
            buf[j] += 0.16 * (p ** 2) * lp

    # ---- cinematic shimmer on the climax (starts ~ last 20 s)
    start = max(starts[-3] if len(starts) >= 3 else 0, 0)
    for f, a in ((329.63, 0.022), (440.0, 0.018), (659.25, 0.010)):
        s0 = int(start * SR)
        for k in range(n - s0):
            j = s0 + k
            t = k / SR
            trem = 0.5 + 0.5 * math.sin(2 * math.pi * 5.5 * t)
            buf[j] += a * trem * math.sin(2 * math.pi * f * t)

    # ---- gentle stereo widen + soft saturation, then normalize
    peak = max(1e-9, max(abs(v) for v in buf))
    g = 0.82 / peak
    out = bytearray()
    for i in range(n):
        v = math.tanh(buf[i] * g * 1.15) * 0.9
        t = i / SR
        # haas-ish width
        w = 0.06 * math.sin(2 * math.pi * 0.13 * t)
        l = int(max(-1.0, min(1.0, v * (1 + w))) * 32000)
        r = int(max(-1.0, min(1.0, v * (1 - w))) * 32000)
        out += struct.pack("<hh", l, r)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(2)
        wf.setsampwidth(2)
        wf.setframerate(SR)
        wf.writeframes(bytes(out))
    print(f"  score: {path} {total:.2f}s")

if __name__ == "__main__":
    build(float(sys.argv[1]), [float(x) for x in sys.argv[2].split(",")])
