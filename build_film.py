#!/usr/bin/env python3
"""Builds "БОЕВОЙ КОНТИНЕНТ" — an epic cinematic short film from generated key art,
Russian voiceover and a synthesized score. Requires ffmpeg (imageio-ffmpeg)."""
import os, subprocess, sys, math

try:
    import imageio_ffmpeg
    FF = imageio_ffmpeg.get_ffmpeg_exe()
except Exception:
    FF = "ffmpeg"

FONT_B = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
FONT_R = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
W, H = 1920, 1080
FPS = 30
OUT = "build"
SEG = os.path.join(OUT, "seg")
os.makedirs(SEG, exist_ok=True)
os.makedirs(os.path.join(OUT, "txt"), exist_ok=True)

def dur(path):
    out = subprocess.run([FF, "-i", path], capture_output=True, text=True).stderr
    for line in out.splitlines():
        if "Duration" in line:
            h, m, s = line.split("Duration:")[1].split(",")[0].strip().split(":")
            return int(h) * 3600 + int(m) * 60 + float(s)
    raise SystemExit("no duration for " + path)

def esc_text(t):
    return t.replace("\\", "\\\\").replace(":", "\\:").replace("'", "\u2019")

def wrap(text, maxc):
    words, lines, cur = text.split(), [], ""
    for w_ in words:
        if len(cur) + len(w_) + 1 <= maxc:
            cur = (cur + " " + w_).strip()
        else:
            lines.append(cur); cur = w_
    if cur:
        lines.append(cur)
    return lines

# ---------------------------------------------------------------- segments
# kind: "scene" (cinemascope crop + ken burns) or "card" (full-frame, poster)
SHOTS = [
    dict(kind="card",  img="assets/poster.png", aud="audio/v01.mp3", zoom="in",
         title="БОЕВОЙ КОНТИНЕНТ", sub="эпическая легенда"),
    dict(kind="scene", img="assets/shot01.png", aud="audio/v02.mp3", zoom="in",
         sub="Там, где облака стали землёй"),
    dict(kind="scene", img="assets/shot02.png", aud="audio/v03.mp3", zoom="out",
         sub="Здесь рождается боевая душа"),
    dict(kind="scene", img="assets/shot03.png", aud="audio/v04.mp3", zoom="in",
         sub="Тан Сан. Сын кузнеца"),
    dict(kind="scene", img="assets/shot07.png", aud="audio/v05.mp3", zoom="in",
         sub="Семь великих сект. Тысячи кланов"),
    dict(kind="scene", img="assets/shot04.png", aud="audio/v06.mp3", zoom="out",
         sub="Континент поднялся против неба"),
    dict(kind="scene", img="assets/shot05.png", aud="audio/v07.mp3", zoom="in",
         sub="Он не просил силы"),
    dict(kind="scene", img="assets/shot04.png", aud="audio/v08.mp3", zoom="in",
         sub="Пока гром не стал его голосом"),
    dict(kind="scene", img="assets/shot06.png", aud="audio/v09.mp3", zoom="out",
         sub="Там, где падали боги"),
    dict(kind="card",  img="assets/poster.png", aud="audio/v10.mp3", zoom="out",
         title="ЛЕГЕНДА НАЧИНАЕТСЯ", sub="БОЕВОЙ КОНТИНЕНТ"),
]

PRE, POST = 0.55, 0.75   # silence before/after each line

def build_segment(i, s):
    ad = dur(s["aud"])
    total = round(ad + PRE + POST, 3)
    frames = int(total * FPS)
    zp_in = f"1+0.00085*on"
    zp_out = f"1.19-0.00085*on"
    z = zp_in if s["zoom"] == "in" else zp_out
    txt_sub = os.path.join(OUT, "txt", f"sub{i}.txt")
    txt_title = os.path.join(OUT, "txt", f"title{i}.txt")
    open(txt_sub, "w").write(esc_text(s["sub"]))

    if s["kind"] == "scene":
        # 2.39:1 cinemascope crop, ken-burns, black bars, grain, subtitle in lower bar
        vf = (
            f"scale=2880:1620:force_original_aspect_ratio=increase,"
            f"crop=2880:1206,"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s=1920x804:fps={FPS},"
            f"eq=contrast=1.06:saturation=1.05:brightness=-0.015,"
            f"unsharp=5:5:0.55:5:5:0.0,"
            f"vignette=PI/5,"
            f"pad={W}:{H}:0:138:black,"
            f"noise=alls=7:allf=t+u,"
            f"drawtext=fontfile={FONT_B}:textfile={txt_sub}:fontcolor=white:fontsize=40"
            f":x=(w-text_w)/2:y={H-92}:borderw=2:bordercolor=black@0.75:alpha=0.95,"
            f"fade=t=in:st=0:d=0.45,fade=t=out:st={max(total-0.45,0):.3f}:d=0.45"
        )
    else:
        txt_title_file = os.path.join(OUT, "txt", f"title{i}.txt")
        open(txt_title_file, "w").write(esc_text(s["title"]))
        vf = (
            f"scale=2200:1240:force_original_aspect_ratio=increase,"
            f"crop=2200:1240,"
            f"zoompan=z='{z}':x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
            f":d={frames}:s={W}x{H}:fps={FPS},"
            f"eq=contrast=1.05:saturation=1.02:brightness=-0.03,"
            f"drawbox=x=0:y=0:w=iw:h=ih:color=black@0.42:t=fill,"
            f"vignette=PI/4.5,"
            f"noise=alls=6:allf=t+u,"
            f"drawtext=fontfile={FONT_B}:textfile={txt_title_file}:fontcolor=white"
            f":fontsize=104:x=(w-text_w)/2:y=(h/2)-90:borderw=3:bordercolor=black@0.8"
            f":shadowx=0:shadowy=6:shadowcolor=black@0.6,"
            f"drawtext=fontfile={FONT_R}:textfile={txt_sub}:fontcolor=0xE8C87A:fontsize=40"
            f":x=(w-text_w)/2:y=(h/2)+56:borderw=2:bordercolor=black@0.7,"
            f"fade=t=in:st=0:d=0.7,fade=t=out:st={max(total-0.7,0):.3f}:d=0.7"
        )

    of = os.path.join(SEG, f"v{i:02d}.mp4")
    if os.path.exists(of) and os.environ.get("REBUILD") != "1":
        print(f"  seg {i:02d} cached  «{s['sub']}»")
        return of, total
    cmd = [FF, "-y", "-loop", "1", "-framerate", str(FPS), "-i", s["img"],
           "-i", s["aud"],
           "-filter_complex",
           f"[0:v]{vf}[v];[1:a]adelay={int(PRE*1000)}|{int(PRE*1000)},"
           f"apad=pad_dur={POST},atrim=0:{total},volume=1.9,"
           f"afade=t=in:st=0:d=0.15,afade=t=out:st={total-0.3:.3f}:d=0.3[a]",
           "-map", "[v]", "-map", "[a]",
           "-t", f"{total}", "-r", str(FPS), "-s", f"{W}x{H}",
           "-c:v", "libx264", "-preset", "medium", "-crf", "19", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-ar", "48000", "-ac", "2", of]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        print(r.stderr[-3000:]); raise SystemExit(f"segment {i} failed")
    print(f"  seg {i:02d} {total:.2f}s  {os.path.basename(s['img'])}  «{s['sub']}»")
    return of, total

print("== rendering segments")
segs, lengths = [], []
for i, s in enumerate(SHOTS):
    f, d = build_segment(i, s)
    segs.append(f); lengths.append(d)
TOTAL = sum(lengths)

# ---------------------------------------------------------------- score
# low drone + rising tension + impact hits, built from sine layers
print("== building score")
def score(out, total, t_seg):
    subprocess.run([sys.executable, "make_score.py", f"{total:.3f}",
                    ",".join(f"{x:.3f}" for x in t_seg)], check=True)

starts, acc = [], 0.0
for d in lengths:
    starts.append(acc); acc += d
score(os.path.join(OUT, "score.wav"), TOTAL, starts)

# ---------------------------------------------------------------- concat
print("== concatenating")
lst = os.path.join(OUT, "list.txt")
with open(lst, "w") as f:
    for s in segs:
        f.write(f"file '{os.path.abspath(s)}'\n")
video = os.path.join(OUT, "video.mp4")
r = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-i", os.path.join(OUT, "score.wav"),
                    "-filter_complex", "[0:a]volume=1.0[va];[1:a]volume=1.0[sa];"
                                       "[va][sa]amix=inputs=2:duration=first:weights=1 0.55,"
                                       "alimiter=limit=0.94[a]",
                    "-map", "0:v", "-map", "[a]", "-c:v", "copy",
                    "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart",
                    os.path.join(OUT, "film_silentmix.mp4")], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("concat failed")

# ---------------------------------------------------------------- credits
print("== end credits")
cred_txt = os.path.join(OUT, "txt", "credits.txt")
open(cred_txt, "w").write(esc_text("Изображения: AI · Голос: Arena · Монтаж: ffmpeg"))
cd = 7.0
vf = (
    f"format=yuv420p,"
    f"drawtext=fontfile={FONT_B}:text='БОЕВОЙ КОНТИНЕНТ':fontcolor=white:fontsize=88"
    f":x=(w-text_w)/2:y=(h/2)-130:alpha='min(1,t/1.2)',"
    f"drawtext=fontfile={FONT_R}:textfile={cred_txt}:fontcolor=0xC9C9C9:fontsize=34"
    f":x=(w-text_w)/2:y=(h/2)+10:alpha='min(1,max(0,(t-1.5)/1.2))',"
    f"drawtext=fontfile={FONT_B}:text='КОНЕЦ ПЕРВОЙ ЧАСТИ':fontcolor=0xE8C87A:fontsize=44"
    f":x=(w-text_w)/2:y=(h/2)+130:alpha='min(1,max(0,(t-3)/1.2))',"
    f"noise=alls=5:allf=t+u,"
    f"fade=t=in:st=0:d=1,fade=t=out:st={cd-1.6}:d=1.6"
)
r = subprocess.run([FF, "-y", "-f", "lavfi", "-i", f"color=c=black:s={W}x{H}:r={FPS}:d={cd}",
                    "-f", "lavfi", "-i", f"anullsrc=r=48000:cl=stereo",
                    "-filter_complex", f"[0:v]{vf}[v]",
                    "-map", "[v]", "-map", "1:a", "-t", f"{cd}",
                    "-c:v", "libx264", "-preset", "medium", "-crf", "19",
                    "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                    os.path.join(OUT, "credits.mp4")], capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("credits failed")

with open(lst, "w") as f:
    for s in segs + [os.path.join(OUT, "credits.mp4")]:
        f.write(f"file '{os.path.abspath(s)}'\n")
final = "БОЕВОЙ_КОНТИНЕНТ.mp4"
r = subprocess.run([FF, "-y", "-f", "concat", "-safe", "0", "-i", lst,
                    "-c", "copy", "-movflags", "+faststart", final],
                   capture_output=True, text=True)
if r.returncode != 0:
    print(r.stderr[-3000:]); raise SystemExit("final concat failed")
print(f"\nГотово: {final}  ({TOTAL + cd:.1f} c)")
