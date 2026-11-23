"""Процедурный эпик-рок движок для песни «Боевой континент».

Всё генерируется из numpy: барабаны, бас, дисторшн-гитары, пэды,
лид-партия, реверб. Никаких сэмплов и внешних библиотек.
"""
import numpy as np

SR = 44100
BPM = 132.0
BEAT = 60.0 / BPM
BAR = 4 * BEAT

_rng = np.random.default_rng(20261123)
_cache = {}


# ---------------------------------------------------------------- утилиты
def hz(m):
    return float(440.0 * 2.0 ** ((m - 69) / 12.0))


def add(buf, sig, t):
    s = int(round(t * SR))
    if s >= len(buf) or s < 0:
        return
    n = min(len(sig), len(buf) - s)
    buf[s:s + n] += sig[:n].astype(buf.dtype)


def _blep(t, dt):
    y = np.zeros_like(t)
    m = t < dt
    if m.any():
        x = t[m] / dt
        y[m] = x + x - x * x - 1.0
    m = t > 1.0 - dt
    if m.any():
        x = (t[m] - 1.0) / dt
        y[m] = x * x + x + x + 1.0
    return y


def saw(freq, n, phase0=0.0):
    dt = freq / SR
    ph = (phase0 + np.arange(n) * dt) % 1.0
    return 2.0 * ph - 1.0 - _blep(ph, dt)


def smooth(x, ms=4.0):
    k = int(max(2, ms * SR / 1000.0))
    w = np.exp(-np.arange(k) / (k / 3.0))
    w /= w.sum()
    return np.convolve(x, w, "same")


# ---------------------------------------------------------------- эквалайзер
def biquad(x, b, a):
    """Прямая форма II, a[0] == 1. Медленно, но без зависимостей."""
    x = np.asarray(x, dtype=np.float64)
    y = np.empty_like(x)
    b0, b1, b2 = b
    _, a1, a2 = a
    x1 = x2 = y1 = y2 = 0.0
    for i in range(len(x)):
        v = b0 * x[i] + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        x2, x1 = x1, x[i]
        y2, y1 = y1, v
        y[i] = v
    return y


def _coefs(b, a):
    return (b[0], b[1], b[2]), (1.0, a[1] / a[0], a[2] / a[0])


def peaking(fc, q, gain_db):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / (2 * q)
    cw = np.cos(w)
    b = np.array([1 + al * A, -2 * cw, 1 - al * A])
    a = np.array([1 + al / A, -2 * cw, 1 - al / A])
    return _coefs(b, a)


def high_shelf(fc, gain_db):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / 2 * np.sqrt(2)
    cw = np.cos(w)
    sq = 2 * np.sqrt(A) * al
    b = A * np.array([(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
                      (A + 1) + (A - 1) * cw - sq])
    a = np.array([(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
                  (A + 1) - (A - 1) * cw - sq])
    return _coefs(b, a)


def low_shelf(fc, gain_db):
    A = 10 ** (gain_db / 40.0)
    w = 2 * np.pi * fc / SR
    al = np.sin(w) / 2 * np.sqrt(2)
    cw = np.cos(w)
    sq = 2 * np.sqrt(A) * al
    b = A * np.array([(A + 1) - (A - 1) * cw + sq, 2 * ((A - 1) - (A + 1) * cw),
                      (A + 1) - (A - 1) * cw - sq])
    a = np.array([(A + 1) + (A - 1) * cw + sq, -2 * ((A - 1) + (A + 1) * cw),
                  (A + 1) + (A - 1) * cw - sq])
    return _coefs(b, a)


def hp1(x, fc):
    """Однополюсный ФВЧ: y = a*(y[-1] + x - x[-1])."""
    a = float(np.exp(-2 * np.pi * fc / SR))
    d = np.diff(np.concatenate([[0.0], np.asarray(x, dtype=np.float64)]))
    y = np.empty_like(d)
    acc = 0.0
    for i in range(len(d)):
        acc = a * acc + a * d[i]
        y[i] = acc
    return y


# ---------------------------------------------------------------- барабаны
def kick_sig(v=1.0):
    n = int(0.32 * SR)
    t = np.arange(n) / SR
    f = 52 + 130 * np.exp(-t * 32)
    ph = 2 * np.pi * np.cumsum(f) / SR
    body = np.sin(ph) * np.exp(-t * 8.0)
    click = _rng.standard_normal(n) * np.exp(-t * 260) * 0.4
    return np.tanh(1.6 * (body + click)) * 0.9 * v


def snare_sig(v=1.0):
    n = int(0.26 * SR)
    t = np.arange(n) / SR
    nz = _rng.standard_normal(n)
    nz = np.diff(np.concatenate([[0.0], nz]))  # грубый highpass
    tone = np.sin(2 * np.pi * 186 * t) * np.exp(-t * 20)
    body = np.sin(2 * np.pi * 330 * t) * np.exp(-t * 45) * 0.3
    y = nz * np.exp(-t * 17) * 0.8 + tone * 0.5 + body
    return np.tanh(1.2 * y) * 0.55 * v


def hat_sig(open_=False, v=1.0):
    key = ("hat", open_, round(v, 3))
    if key in _cache:
        return _cache[key]
    dur = 0.32 if open_ else 0.06
    n = int(dur * SR)
    t = np.arange(n) / SR
    nz = _rng.standard_normal(n)
    hp = np.diff(np.concatenate([[0.0], nz]))
    y = hp * np.exp(-t * (10 if open_ else 70)) * 0.42
    _cache[key] = y * v
    return _cache[key]


def crash_sig(v=1.0):
    n = int(1.9 * SR)
    t = np.arange(n) / SR
    nz = _rng.standard_normal(n)
    hp = np.diff(np.concatenate([[0.0], np.diff(np.concatenate([[0.0], nz]))]))
    metal = np.sign(np.sin(2 * np.pi * 3170 * t)) * 0.02 * np.exp(-t * 4)
    return (hp * np.exp(-t * 2.5) * 0.20 + metal) * v


def tom_sig(f0=110.0, v=1.0):
    n = int(0.38 * SR)
    t = np.arange(n) / SR
    f = f0 * (0.72 + 0.28 * np.exp(-t * 14))
    ph = 2 * np.pi * np.cumsum(f) / SR
    y = np.sin(ph) * np.exp(-t * 9)
    return np.tanh(1.4 * y) * 0.7 * v


# ---------------------------------------------------------------- инструменты
def power_chord_sig(root_midi, dur, drive=4.5, v=1.0, short=0.45):
    key = ("gtr", root_midi, round(dur, 4), round(v, 3), short)
    if key in _cache:
        return _cache[key]
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    f0 = hz(root_midi)
    x = np.zeros(n)
    for mult, amp in ((1, 1.0), (1.4983, 0.85), (2, 0.72), (2.997, 0.46), (4.02, 0.28), (5.1, 0.12)):
        x += amp * saw(f0 * mult, n, phase0=_rng.random())
    x /= 2.9
    y = np.tanh(drive * x)
    att = np.minimum(1.0, t / 0.004)
    env = att * (0.42 + 0.58 * np.exp(-t / max(short, 0.05)))
    rel = int(min(0.02, dur) * SR)
    if rel > 1:
        env[-rel:] *= np.linspace(1.0, 0.0, rel)
    _cache[key] = (y * env * 0.30 * v).astype(np.float64)
    return _cache[key]


def bass_sig(root_midi, dur, v=1.0):
    key = ("bass", root_midi, round(dur, 4), round(v, 3))
    if key in _cache:
        return _cache[key]
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    f = hz(root_midi)
    x = (np.sin(2 * np.pi * f * t) * 0.62 + saw(f, n) * 0.30
         + np.sin(4 * np.pi * f * t) * 0.20 + saw(2 * f, n) * 0.10)
    env = np.minimum(1.0, t / 0.006) * (0.55 + 0.45 * np.exp(-t * 3.5))
    rel = int(min(0.015, dur) * SR)
    if rel > 1:
        env[-rel:] *= np.linspace(1.0, 0.0, rel)
    y = np.tanh(1.35 * x) * env * 0.34 * v
    _cache[key] = y
    return _cache[key]


def lead_sig(midi, dur, v=1.0, octave=False):
    key = ("lead", midi, round(dur, 4), round(v, 3), octave)
    if key in _cache:
        return _cache[key]
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    f = hz(midi) * (2 if octave else 1)
    vib = 1.0 + 0.011 * np.sin(2 * np.pi * 5.4 * t) * np.minimum(1.0, t / 0.3)
    ph = 2 * np.pi * np.cumsum(f * vib) / SR
    y = (np.sin(ph) + 0.42 * np.sin(2 * ph) + 0.26 * np.sin(3 * ph)
         + 0.14 * np.sin(4 * ph) + 0.07 * np.sin(5 * ph))
    y += 0.25 * np.sin(ph * 1.004)  # лёгкий расстроенный дубль
    env = np.minimum(1.0, t / 0.02) * (0.75 + 0.25 * np.exp(-t * 1.5))
    rel = int(min(0.10, dur * 0.5) * SR)
    if rel > 1:
        env[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.4
    y = y * env * 0.15 * v
    _cache[key] = y
    return _cache[key]


def pad_sig(midis, dur, v=1.0):
    key = ("pad", tuple(midis), round(dur, 3), round(v, 3))
    if key in _cache:
        return _cache[key]
    n = max(8, int(dur * SR))
    t = np.arange(n) / SR
    x = np.zeros(n)
    for m in midis:
        f = hz(m)
        for det in (-0.005, 0.0, 0.006):
            x += saw(f * (1 + det), n, phase0=_rng.random()) * 0.15
    x /= max(1, len(midis))
    x = smooth(x, 3.0)
    a = 0.30
    env = np.minimum(1.0, t / a)
    rel = int(min(0.45, dur * 0.45) * SR)
    if rel > 1:
        env[-rel:] *= np.linspace(1.0, 0.0, rel) ** 1.3
    y = np.tanh(0.8 * x) * env * 0.17 * v
    _cache[key] = y
    return _cache[key]


# ---------------------------------------------------------------- гармония
ROOT = {"Dm": 38, "Bb": 34, "F": 41, "C": 36, "Gm": 43, "A": 45}
TRIAD = {
    "Dm": (50, 53, 57),
    "Bb": (46, 53, 58),
    "F": (53, 57, 60),
    "C": (48, 52, 55),
    "Gm": (43, 46, 50),
    "A": (45, 49, 52),
}
PROG = {
    "intro": ["Dm", "Dm", "Bb", "Bb"],
    "title": ["Dm", "Dm", "Bb", "C"],
    "verse": ["Dm", "Bb", "F", "C"],
    "pre": ["Bb", "C", "Dm", "A"],
    "chorus": ["Bb", "C", "Dm", "A"],
    "interlude": ["Bb", "C", "Dm", "Dm"],
    "bridge": ["Dm", "Bb", "Gm", "A"],
    "outro": ["Dm", "Bb", "F", "C"],
}


def chord_at(section, bar):
    p = PROG[section]
    return p[bar % len(p)]


# ---------------------------------------------------------------- аранжировка
def drum_bar(stems, t0, style, bar_i=0, big=False):
    if style == "none":
        return
    if style == "verse":
        add(stems["kick"], kick_sig(0.95), t0)
        add(stems["kick"], kick_sig(0.8), t0 + 2 * BEAT)
        add(stems["kick"], kick_sig(0.7), t0 + 3.5 * BEAT) if bar_i % 2 else None
        add(stems["snare"], snare_sig(0.85), t0 + BEAT)
        add(stems["snare"], snare_sig(0.95), t0 + 3 * BEAT)
        for i in range(8):
            add(stems["hat"], hat_sig(False, 0.55 if i % 2 else 0.8), t0 + i * BEAT / 2)
        if bar_i % 4 == 0:
            add(stems["crash"], crash_sig(0.55), t0)
    elif style == "pre":
        add(stems["kick"], kick_sig(0.9), t0)
        add(stems["kick"], kick_sig(0.85), t0 + 2 * BEAT)
        add(stems["snare"], snare_sig(0.9), t0 + BEAT)
        add(stems["snare"], snare_sig(0.95), t0 + 3 * BEAT)
        for i in range(8):
            add(stems["hat"], hat_sig(False, 0.5 + 0.3 * (i % 2)), t0 + i * BEAT / 2)
    elif style == "chorus":
        add(stems["kick"], kick_sig(1.0), t0)
        add(stems["kick"], kick_sig(0.85), t0 + 1.5 * BEAT)
        add(stems["kick"], kick_sig(0.95), t0 + 2 * BEAT)
        if bar_i % 2 == 1:
            add(stems["kick"], kick_sig(0.7), t0 + 3.5 * BEAT)
        add(stems["snare"], snare_sig(1.0), t0 + BEAT)
        add(stems["snare"], snare_sig(1.0), t0 + 3 * BEAT)
        for i in range(8):
            add(stems["hat"], hat_sig(False, 0.45 + 0.35 * (i % 2)), t0 + i * BEAT / 2)
        add(stems["hat"], hat_sig(True, 0.5), t0 + 3.5 * BEAT)
        add(stems["crash"], crash_sig(0.85 if big else 0.7), t0 + (0 if bar_i % 2 == 0 else 2 * BEAT))
    elif style == "interlude":
        for i in range(8):
            add(stems["kick"], kick_sig(0.85 if i % 2 == 0 else 0.6), t0 + i * BEAT / 2)
        add(stems["snare"], snare_sig(0.95), t0 + BEAT)
        add(stems["snare"], snare_sig(0.95), t0 + 3 * BEAT)
        for i in range(8):
            add(stems["hat"], hat_sig(False, 0.6), t0 + i * BEAT / 2)
        if bar_i % 2 == 0:
            add(stems["crash"], crash_sig(0.8), t0)
    elif style == "bridge":
        add(stems["kick"], kick_sig(1.0), t0)
        add(stems["kick"], kick_sig(0.85), t0 + 2.5 * BEAT)
        add(stems["snare"], snare_sig(1.1), t0 + 2 * BEAT)
        add(stems["snare"], snare_sig(0.6), t0 + 3.25 * BEAT)
        add(stems["tom"], tom_sig(90, 0.9), t0 + 3.5 * BEAT)
        for i in range(4):
            add(stems["hat"], hat_sig(False, 0.5), t0 + i * BEAT)
        if bar_i % 2 == 0:
            add(stems["crash"], crash_sig(0.9), t0)
    elif style == "outro":
        if bar_i < 2:
            add(stems["kick"], kick_sig(0.9), t0)
            add(stems["snare"], snare_sig(0.8), t0 + 2 * BEAT)
            add(stems["hat"], hat_sig(False, 0.4), t0 + 2 * BEAT)
            if bar_i == 0:
                add(stems["crash"], crash_sig(0.8), t0)


def fill(stems, t0, kind="tom"):
    """Филл в конце секции (последняя доля такта)."""
    if kind == "tom":
        for i, f in enumerate((180, 150, 120, 95)):
            add(stems["tom"], tom_sig(f, 0.5 + 0.15 * i), t0 + 3 * BEAT + i * BEAT / 4)
    elif kind == "snare":
        for i in range(8):
            add(stems["snare"], snare_sig(0.35 + 0.09 * i), t0 + 2 * BEAT + i * BEAT / 4)


def riser(stems, t0, dur):
    n = int(dur * SR)
    t = np.arange(n) / SR
    nz = _rng.standard_normal(n)
    nz = np.diff(np.concatenate([[0.0], nz]))
    y = nz * (t / dur) ** 2.2 * 0.18
    add(stems["fx"], y, t0)


def section(stems, t0, bars, kind, big=False):
    """Отрисовывает инструментал секции (без вокала)."""
    style = {
        "intro": "none", "title": "none", "verse": "verse", "pre": "pre",
        "chorus": "chorus", "interlude": "interlude", "bridge": "bridge",
        "outro": "outro",
    }[kind]

    for b in range(bars):
        tb = t0 + b * BAR
        ch = chord_at(kind, b)

        # --- барабаны
        if kind == "intro":
            if b >= 1:
                add(stems["tom"], tom_sig(120, 0.35), tb)
                add(stems["tom"], tom_sig(120, 0.3), tb + 2 * BEAT)
        elif kind == "title":
            add(stems["kick"], kick_sig(0.85 if b == 0 else 0.6), tb)
            add(stems["snare"], snare_sig(0.55 if b == 0 else 0.4), tb + 2 * BEAT)
            add(stems["tom"], tom_sig(95, 0.45), tb + 3.5 * BEAT)
            if b == 0:
                add(stems["crash"], crash_sig(0.7), tb)
            if b >= 2:
                add(stems["crash"], crash_sig(0.35), tb)
        elif kind == "outro":
            drum_bar(stems, tb, "outro", b)
        else:
            drum_bar(stems, tb, style, b, big)

        # --- бас
        if kind in ("intro", "outro"):
            if b % 2 == 0:
                add(stems["bass"], bass_sig(ROOT[ch], BAR * 1.9, 0.7), tb)
        elif kind == "title":
            add(stems["bass"], bass_sig(ROOT[ch], BAR * 0.95, 0.7), tb)
        else:
            for i in range(8):
                add(stems["bass"], bass_sig(ROOT[ch], BEAT * 0.46,
                                            0.9 if i % 2 == 0 else 0.72), tb + i * BEAT / 2)
            if kind == "bridge":
                for i in range(8):
                    add(stems["bass"], bass_sig(ROOT[ch] + 12, BEAT * 0.5, 0.55),
                        tb + i * BEAT / 2)

        # --- гитары
        if kind in ("verse", "pre", "chorus", "interlude"):
            base = {"verse": 0.78, "pre": 0.88, "chorus": 1.05, "interlude": 1.0}[kind]
            for i in range(8):
                v = base * (1.0 if i % 2 == 0 else 0.78)
                d = BEAT * 0.46
                add(stems["gtrL"], power_chord_sig(ROOT[ch], d, 4.5, v, 0.30),
                    tb + i * BEAT / 2)
                add(stems["gtrR"], power_chord_sig(ROOT[ch], d, 4.2, v * 0.95, 0.34),
                    tb + i * BEAT / 2 + 0.004)
        elif kind == "chorus" and big:
            for i in range(4):
                add(stems["lead"], lead_sig(ROOT[ch] + 24, BEAT * 0.9, 0.22), tb + i * BEAT)
        if kind == "chorus":
            for i in range(4):
                add(stems["gtrL"], power_chord_sig(ROOT[ch] + 12, BEAT * 0.94, 3.0, 0.32, 0.5),
                    tb + i * BEAT)
                add(stems["gtrR"], power_chord_sig(ROOT[ch] + 12, BEAT * 0.94, 2.8, 0.30, 0.5),
                    tb + i * BEAT + 0.006)
        if kind == "bridge":
            add(stems["gtrL"], power_chord_sig(ROOT[ch], BAR * 0.95, 5.0, 0.95, 2.2), tb)
            add(stems["gtrR"], power_chord_sig(ROOT[ch], BAR * 0.95, 4.6, 0.9, 2.4), tb + 0.006)
        elif kind == "outro":
            add(stems["gtrL"], power_chord_sig(ROOT[ch], BAR * 0.95, 5.0, 0.85, 2.5), tb)
            add(stems["gtrR"], power_chord_sig(ROOT[ch], BAR * 0.95, 4.6, 0.8, 2.7), tb + 0.006)
        elif kind == "title":
            for i in range(4):
                v = 0.55 if b < 2 else 0.85
                add(stems["gtrL"], power_chord_sig(ROOT[ch], BEAT * 0.92, 3.6, v, 0.8), tb + i * BEAT)
                add(stems["gtrR"], power_chord_sig(ROOT[ch], BEAT * 0.92, 3.4, v * 0.95, 0.9),
                    tb + i * BEAT + 0.006)
        elif kind == "intro":
            if b >= 2:
                add(stems["gtrL"], power_chord_sig(ROOT[ch], BAR * 0.95, 3.5, 0.55, 2.0), tb)
                add(stems["gtrR"], power_chord_sig(ROOT[ch], BAR * 0.95, 3.2, 0.5, 2.2), tb + 0.006)

        # --- пэд
        if kind in ("intro", "title", "verse", "pre", "chorus", "interlude", "bridge", "outro"):
            dur = BAR * (1.9 if kind in ("intro", "title", "outro") and b % 2 == 0 else 0.98)
            add(stems["pad"], pad_sig(TRIAD[ch], dur, 0.9 if kind != "bridge" else 1.0), tb)

        # --- лид
        if kind == "intro":
            for (bb, bt, d, m) in INTRO_MOTIF:
                if bb == b:
                    add(stems["lead"], lead_sig(m, d * BEAT, 0.95), tb + bt * BEAT)
        elif kind == "title":
            if b == 0:
                add(stems["lead"], lead_sig(74, BAR * 1.8, 0.6), tb)
        elif kind in ("chorus", "interlude"):
            for (bb, bt, d, m) in CHORUS_HOOK:
                if bb == b % 4:
                    add(stems["lead"], lead_sig(m, d * BEAT, 1.0 if kind == "interlude" else 0.85),
                        tb + bt * BEAT)
                    if kind == "chorus" and big:
                        add(stems["lead"], lead_sig(m, d * BEAT, 0.35, octave=True), tb + bt * BEAT)
        elif kind == "outro":
            for (bb, bt, d, m) in OUTRO_MOTIF:
                if bb == b:
                    add(stems["lead"], lead_sig(m, d * BEAT, 0.8), tb + bt * BEAT)

    # --- переходы
    if kind == "pre" and bars >= 2:
        fill(stems, t0 + (bars - 2) * BAR, "snare")
        fill(stems, t0 + (bars - 1) * BAR, "tom")
        riser(stems, t0 + (bars - 1) * BAR, BAR)
    if kind == "intro":
        riser(stems, t0 + (bars - 1) * BAR, BAR)
    if kind == "verse" and bars >= 2:
        fill(stems, t0 + (bars - 1) * BAR, "tom")
    if kind == "interlude":
        fill(stems, t0 + (bars - 1) * BAR, "tom")
        riser(stems, t0 + (bars - 1) * BAR, BAR)
    if kind == "bridge" and bars >= 2:
        fill(stems, t0 + (bars - 1) * BAR, "tom")


INTRO_MOTIF = [
    (0, 0.0, 2.0, 74), (0, 2.5, 1.5, 77),
    (1, 0.0, 1.0, 76), (1, 1.0, 1.0, 74), (1, 2.5, 1.5, 72),
    (2, 0.0, 2.0, 69), (2, 2.0, 1.0, 72), (3, 0.0, 3.0, 74),
]
CHORUS_HOOK = [
    (0, 0.0, 1.5, 77), (0, 1.5, 0.5, 79), (0, 2.0, 2.0, 81),
    (1, 0.0, 2.0, 81), (1, 2.0, 1.0, 79), (1, 3.0, 1.0, 77),
    (2, 0.0, 1.5, 76), (2, 1.5, 0.5, 74), (2, 2.0, 2.0, 72),
    (3, 0.0, 1.0, 73), (3, 1.0, 1.0, 76), (3, 2.0, 2.0, 74),
]
OUTRO_MOTIF = [
    (0, 0.0, 3.0, 74), (1, 0.0, 3.0, 77),
    (2, 0.0, 1.5, 76), (2, 1.5, 1.5, 74), (3, 0.0, 4.0, 74),
]


# ---------------------------------------------------------------- сведение
def make_ir(seed, length=2.4, decay=2.6):
    rng = np.random.default_rng(seed)
    n = int(length * SR)
    t = np.arange(n) / SR
    x = rng.standard_normal(n) * np.exp(-t * decay)
    x = smooth(x, 6.0)
    x[: int(0.018 * SR)] = 0
    return x / np.sqrt(np.sum(x ** 2))


def fft_convolve(x, h):
    L = len(h)
    B = 1
    while B < L:
        B <<= 1
    nfft = 1 << int(np.ceil(np.log2(B + L - 1)))
    H = np.fft.rfft(h, nfft)
    out = np.zeros(len(x) + L - 1, dtype=np.float64)
    for s in range(0, len(x), B):
        blk = x[s:s + B]
        if len(blk) == 0:
            break
        Y = np.fft.irfft(np.fft.rfft(blk, nfft) * H, nfft)
        e = min(len(out), s + nfft)
        out[s:e] += Y[: e - s]
    return out[: len(x)]


def haas(x, ms=7.0):
    d = int(ms * SR / 1000.0)
    y = np.concatenate([np.zeros(d), x[:-d]]) if d < len(x) else np.zeros_like(x)
    return x * 0.72 + y * 0.55


def mixdown(stems, n_total, vocal=None, vocal_l=None, vocal_r=None, chant=None):
    """stems: dict mono float32; vocal: mono float32 уже обработанный."""
    L = np.zeros(n_total, dtype=np.float64)
    R = np.zeros(n_total, dtype=np.float64)

    def place(sig, pl, pr):
        nonlocal L, R
        n = min(len(sig), n_total)
        L[:n] += sig[:n] * pl
        R[:n] += sig[:n] * pr

    place(stems["kick"], 0.70, 0.70)
    place(stems["snare"], 0.92, 0.92)
    place(stems["hat"], 0.66, 0.90)
    place(stems["tom"], 0.8, 0.8)
    place(stems["crash"], 0.55, 0.85)
    place(stems["bass"], 0.66, 0.66)
    place(stems["gtrL"], 1.05, 0.46)
    place(stems["gtrR"], 0.46, 1.05)
    place(haas(stems["pad"], 6.0), 0.9, 0.55)
    place(stems["lead"], 0.62, 0.8)
    place(stems["fx"], 0.4, 0.7)

    if vocal is not None:
        place(vocal, 1.05, 1.05)
    if vocal_l is not None:   # дабл-трек, влево
        place(vocal_l, 0.34, 0.12)
    if vocal_r is not None:   # эхо-дилей, вправо
        place(vocal_r, 0.10, 0.30)
    if chant is not None:     # толпа, «эй!»
        place(chant, 0.5, 0.55)

    # --- реверб-посыл
    send = (stems["pad"] * 0.55 + stems["lead"] * 0.6 + stems["snare"] * 0.5
            + stems["tom"] * 0.35 + stems["crash"] * 0.3
            + (stems["gtrL"] + stems["gtrR"]) * 0.18)
    if vocal is not None:
        send = send + vocal * 0.42
    if vocal_l is not None:
        send = send + vocal_l * 0.5
    send = send.astype(np.float64)
    rvL = fft_convolve(send, make_ir(11))
    rvR = fft_convolve(send, make_ir(29))
    L += rvL * 0.16
    R += rvR * 0.16

    # --- мастер-EQ: убираем саб-гул и грязь, добавляем презенс
    mono_stereo = [L, R]
    for i, ch in enumerate(mono_stereo):
        ch = hp1(ch, 32.0)
        ch = biquad(ch, *low_shelf(90.0, -2.5))
        ch = biquad(ch, *peaking(280.0, 0.9, -2.0))
        ch = biquad(ch, *peaking(390.0, 1.2, -1.2))
        ch = biquad(ch, *peaking(1800.0, 1.1, 1.5))
        ch = biquad(ch, *high_shelf(4200.0, 3.0))
        mono_stereo[i] = ch
    L, R = mono_stereo

    # --- лимитер/нормализация
    mx = max(np.max(np.abs(L)), np.max(np.abs(R)), 1e-9)
    L = np.tanh(L / mx * 1.25)
    R = np.tanh(R / mx * 1.25)
    pk = max(np.max(np.abs(L)), np.max(np.abs(R)))
    L *= 0.96 / pk
    R *= 0.96 / pk
    return np.stack([L.astype(np.float32), R.astype(np.float32)], axis=1)
