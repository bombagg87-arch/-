#!/usr/bin/env python3
"""
Сборка «одного файла» фильма «Грозовой клинок».

Берёт film/index.html и вклеивает внутрь все картинки и аудио как data:URI,
чтобы получился один самодостаточный .html, который открывается без сервера
и без интернета.

Запуск:  python3 film/build_single.py
Выход:   grozovoy-klinok-film.html (в корне репозитория)
"""
import base64
import mimetypes
import pathlib
import re
import sys

ROOT = pathlib.Path(__file__).resolve().parent
SRC = ROOT / "index.html"
OUT = ROOT.parent / "grozovoy-klinok-film.html"


def data_uri(path: pathlib.Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def main() -> int:
    html = SRC.read_text(encoding="utf-8")

    # 1) кадры и дорожки объявлены в массиве SCENES
    assets = sorted({m for m in re.findall(r'(?:img|audio):\s*"((?:img|audio)/[^"]+)"', html)})
    if not assets:
        print("Не нашёл ссылок на ассеты — изменилась структура index.html", file=sys.stderr)
        return 1

    for rel in assets:
        f = ROOT / rel
        if not f.exists():
            print(f"Нет файла: {f}", file=sys.stderr)
            return 1
        html = html.replace(f'"{rel}"', '"' + data_uri(f) + '"')
        print(f"  вклеен {rel:26s} {f.stat().st_size / 1024:7.0f} КБ")

    # 2) фоновый постер на заставке (CSS)
    poster = data_uri(ROOT / "img" / "poster.jpg")
    html = re.sub(r'url\("img/poster\.jpg"\)', f'url("{poster}")', html)

    # 3) favicon: отдельный файл больше не нужен
    html = re.sub(r'\s*<link rel="icon"[^>]*>', "", html)

    # 4) контроль: никаких относительных путей к ассетам не осталось
    leftovers = re.findall(r'(?:url\("|src="|href=")((?:img|audio)/[^"]+)"', html)
    if leftovers:
        print("Остались относительные ссылки:", leftovers, file=sys.stderr)
        return 1

    # 5) медиа из data:URI -> blob:URL (надёжнее для перемотки и старых Safari)
    anchor = "const $ = (s) => document.querySelector(s);"
    if anchor not in html:
        print("Не нашёл точку вставки для blob-конвертера", file=sys.stderr)
        return 1
    blob_shim = """/* автономный режим: превращаем встроенные data:URI в blob:URL */
(function toBlobs(){
  const toURL = (src) => {
    if (typeof src !== "string" || src.indexOf("data:") !== 0) return src;
    try {
      const comma = src.indexOf(",");
      const mime = src.slice(5, src.indexOf(";", comma));
      const bin = atob(src.slice(comma + 1));
      const arr = new Uint8Array(bin.length);
      for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
      return URL.createObjectURL(new Blob([arr], { type: mime }));
    } catch (e) { return src; }
  };
  SCENES.forEach(s => { s.img = toURL(s.img); s.audio = toURL(s.audio); });
})();

"""
    html = html.replace(anchor, blob_shim + anchor, 1)

    # 6) помечаем, что это автономная версия
    html = html.replace(
        "<title>ГРОЗОВОЙ КЛИНОК — фильм с Боевого континента</title>",
        "<title>ГРОЗОВОЙ КЛИНОК — фильм с Боевого континента</title>\n"
        "<!-- автономная версия: все кадры и озвучка встроены в файл -->",
    )

    OUT.write_text(html, encoding="utf-8")
    print(f"\nГотово: {OUT}  ({OUT.stat().st_size / 1024 / 1024:.1f} МБ)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
