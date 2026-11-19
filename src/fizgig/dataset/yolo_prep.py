"""Fork-only (branch custom): every picture plus a close-up of each YOLO detection, each with its -masklabel.png.

CLI:  python yolo_prep.py <yolo.pt> <input folder> <output folder> [--pad 20]
Web:  start(folder, model, pad) - Image Prep's masks card, Custom YOLO (static/yolo_prep.js). The pictures come from
      the training folder's 'originals' subfolder (made from the folder's pictures the first time); the training
      folder's pictures, captions and masks are deleted and made again from it on every run.
"""
import argparse
import importlib.util
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

EXTS = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
MASK = "-masklabel"
MIN_COVER = 0.01  # Fizgig ignores a mask with less white than this


def crop_box(box, pad, size):
    """Fizgig's crop_to_face: pad the box, make it square, clip to the picture."""
    x1, y1, x2, y2 = (int(v) for v in box)
    pw, ph = int((x2 - x1) * pad / 100), int((y2 - y1) * pad / 100)
    x1, y1, x2, y2 = x1 - pw, y1 - ph, x2 + pw, y2 + ph
    w, h = x2 - x1, y2 - y1
    if w > h:
        y1 -= (w - h) // 2
        y2 = y1 + w
    else:
        x1 -= (h - w) // 2
        x2 = x1 + h
    return max(0, x1), max(0, y1), min(size[0], x2), min(size[1], y2)


def main():
    import numpy as np
    from PIL import Image, ImageOps
    from ultralytics import YOLO

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("model", help="YOLO .pt (segm: outline masks, bbox: box masks)")
    ap.add_argument("src", type=Path, help="folder of pictures (read only)")
    ap.add_argument("dst", type=Path, help="output folder = Fizgig training folder")
    ap.add_argument("--pad", type=float, default=20.0, help="close-up padding around the box, percent (default 20)")
    a = ap.parse_args()
    if a.dst.resolve() == a.src.resolve():
        ap.error("output folder must differ from the input folder")

    model = YOLO(a.model)
    a.dst.mkdir(parents=True, exist_ok=True)
    files = [p for p in sorted(a.src.iterdir()) if p.suffix.lower() in EXTS and not p.stem.lower().endswith(MASK)]
    crops = 0
    for n, p in enumerate(files, 1):
        name = f"{n:04d}"
        print(f"{name} <- {p.name}")
        img = ImageOps.exif_transpose(Image.open(p)).convert("RGB")
        img.save(a.dst / f"{name}_0.png")

        r = model(img, retina_masks=True, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy()
        if not len(boxes):
            print(f"[no detection] {name} {p.name}")
            continue

        mask = np.zeros((img.height, img.width), np.uint8)
        if r.masks is not None:
            mask[(r.masks.data.cpu().numpy() > 0.5).any(0)] = 255
        else:
            for x1, y1, x2, y2 in boxes.astype(int):
                mask[y1:y2, x1:x2] = 255
        cover = mask.mean() / 255
        if cover < MIN_COVER:
            print(f"[<{MIN_COVER:.0%}] {name} {p.name}: mask covers {cover:.2%}, Fizgig will train this picture unmasked")
        mask = Image.fromarray(mask)
        mask.save(a.dst / f"{name}_0{MASK}.png")

        for i, box in enumerate(boxes, 1):
            c = crop_box(box, a.pad, img.size)
            img.crop(c).save(a.dst / f"{name}_{i}.png")
            mask.crop(c).save(a.dst / f"{name}_{i}{MASK}.png")
            crops += 1

    print(f"Pictures: {len(files)} | Close-ups: {crops} | Output: {a.dst}")


# ---- the web: a prep run on the training folder (the same job, log and state as prep_jobs) ------------------------
def _pictures(folder):
    return [f for f in sorted(os.listdir(folder)) if os.path.isfile(os.path.join(folder, f))
            and os.path.splitext(f)[1].lower() in EXTS and not os.path.splitext(f)[0].lower().endswith(MASK)]


def start(folder, model, pad=20):
    from fizgig.web import prep_jobs as pj
    if not folder or not os.path.isdir(folder):
        raise ValueError("Choose the training folder on the Start tab first.")
    if not model or not os.path.isfile(model):
        raise ValueError("Type the full path of your YOLO .pt file.")
    if importlib.util.find_spec("ultralytics") is None:
        raise ValueError(f"Custom YOLO needs ultralytics in Fizgig's venv: {sys.executable} -m pip install ultralytics")
    pad = float(pad)
    with pj._lock:
        if pj._job["state"] == "running":
            raise ValueError("A prep run is already going.")
        if pj._look["state"] == "running":
            raise ValueError("The Look Filter is scanning. Wait for it to finish.")
        pj._job.update(state="running", log="", folder=folder, started=time.time())

    def work():
        try:
            src = os.path.join(folder, "originals")
            if not os.path.isdir(src):
                os.makedirs(src)
                pics = _pictures(folder)
                for f in pics:
                    shutil.copy2(os.path.join(folder, f), os.path.join(src, f))
                pj._log(f"Copied {len(pics)} picture(s) to originals\n")
            if not _pictures(src):
                raise ValueError("no pictures in the originals subfolder")
            gone = 0
            for f in os.listdir(folder):
                p = os.path.join(folder, f)
                if os.path.isfile(p) and os.path.splitext(f)[1].lower() in EXTS | {".txt"}:
                    os.remove(p)
                    gone += 1
            pj._log(f"Deleted {gone} old picture(s), caption(s) and mask(s) from the training folder\n\n")
            proc = subprocess.Popen([sys.executable, "-u", os.path.abspath(__file__), model, src, folder,
                                     "--pad", str(pad)],
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"},
                                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
            for line in proc.stdout:
                pj._log(line)
            if proc.wait():
                raise RuntimeError(f"yolo_prep exited with code {proc.returncode}")
            pj._log("\nDone. Make the captions again on the Captions tab.\n")
            end = "done"
        except Exception as e:
            pj._log(f"\nERROR: Custom YOLO failed — {type(e).__name__}: {e}\n")
            end = "failed"
        with pj._lock:
            pj._job["state"] = end
    threading.Thread(target=work, daemon=True).start()
    return {"ok": True}


if __name__ == "__main__":
    main()
