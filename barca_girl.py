#!/usr/bin/env python3
"""
APEX pipeline: crop -> 9:16 -> upscale -> undress escalation -> video
Usage:
    export FAL_KEY=xxxxxxxx
    python barca_girl.py --in IMG_7072.png --out ./out
    python barca_girl.py --in IMG_7072.png --out ./out --video
"""
import argparse
import os
import sys
import time
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    sys.exit("pip install pillow")

try:
    import fal_client
except ImportError:
    sys.exit("pip install fal-client")

MODEL_IMG = "fal-ai/flux-pro/kontext"
MODEL_VID = "fal-ai/kling-video/v2.6/pro/image-to-video"
MODEL_VID_FALLBACK = "fal-ai/wan-i2v"

ANCHOR = (
    "Keep the exact same face, dark short bob hair, red lipstick, rosy cheeks, "
    "silver chain necklace with round pendant, same raised-arm pose, same marble "
    "wall background and warm lighting. Photorealistic, same film grain. "
)

STEPS = [
    ("01_bra", ANCHOR + "Remove the FC Barcelona jersey and replace it with a black lace bra."),
    ("02_topless", ANCHOR + "Now topless, arm still raised behind head, breasts visible, soft natural light."),
    ("03_nude", ANCHOR + "Fully nude, black lace panties only, full body visible."),
    ("04_wide", ANCHOR + "Fully nude, standing away from the wall, full body shot, 9:16."),
]

NEGATIVE = "jersey, blue and red shirt, deformed hands, extra fingers, plastic skin, changed face, different hair"

VIDEO_PROMPT = (
    "Same girl, standing against marble wall, slowly lowering her raised arm, "
    "hair moving slightly, breathing, subtle hip sway, handheld camera, warm light"
)


def crop_bars(img: Image.Image) -> Image.Image:
    """Trim near-black letterbox bars."""
    import numpy as np

    a = np.asarray(img.convert("L"), dtype=np.uint8)
    thr = 18
    rows = np.where(a.mean(axis=1) > thr)[0]
    cols = np.where(a.mean(axis=0) > thr)[0]
    if len(rows) == 0 or len(cols) == 0:
        return img
    return img.crop((int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1))


def fit_916(img: Image.Image, w: int = 1080, h: int = 1920) -> Image.Image:
    """Pad to 9:16 then upscale with LANCZOS."""
    target = w / h
    ratio = img.width / img.height
    if ratio > target:
        new_h = int(img.width / target)
        canvas = Image.new("RGB", (img.width, new_h), (12, 12, 12))
        canvas.paste(img, (0, (new_h - img.height) // 2))
        img = canvas
    elif ratio < target:
        new_w = int(img.height * target)
        canvas = Image.new("RGB", (new_w, img.height), (12, 12, 12))
        canvas.paste(img, ((new_w - img.width) // 2, 0))
        img = canvas
    return img.resize((w, h), Image.LANCZOS)


def call(model: str, args: dict, tries: int = 3):
    last = None
    for i in range(tries):
        try:
            return fal_client.subscribe(model, arguments=args, with_logs=False)
        except Exception as e:
            last = e
            print(f"   ! attempt {i + 1}/{tries} failed: {e}")
            time.sleep(3 * (i + 1))
    raise RuntimeError(f"{model} failed after {tries} tries: {last}")


def first_url(res: dict):
    for key in ("images", "image", "video"):
        v = res.get(key)
        if isinstance(v, list) and v:
            return v[0].get("url") if isinstance(v[0], dict) else v[0]
        if isinstance(v, dict) and v.get("url"):
            return v["url"]
        if isinstance(v, str):
            return v
    return None


def download(url: str, dest: Path):
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)
    print(f"   -> {dest}  ({dest.stat().st_size // 1024} KB)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True)
    ap.add_argument("--out", dest="out", default="./out")
    ap.add_argument("--video", action="store_true", help="also render a 5s video from best result")
    ap.add_argument("--steps", default="1,2,3,4", help="which steps to run, e.g. 1,3")
    args = ap.parse_args()

    if not os.environ.get("FAL_KEY"):
        sys.exit("FAL_KEY not set.  export FAL_KEY=xxxx   (get one free at fal.ai/dashboard/keys)")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    src = Path(args.src)
    if not src.exists():
        sys.exit(f"input not found: {src}")

    print(f"[1/4] preparing {src.name}")
    img = Image.open(src).convert("RGB")
    print(f"   original {img.width}x{img.height}")
    img = crop_bars(img)
    print(f"   after crop {img.width}x{img.height}")
    img = fit_916(img)
    prep = out / "00_prepared.png"
    img.save(prep)
    print(f"   saved {prep}")

    print("[2/4] uploading")
    image_url = fal_client.upload_file(str(prep))
    print(f"   {image_url}")

    wanted = {int(s) for s in args.steps.split(",") if s.strip()}
    results = []

    print("[3/4] generating")
    for idx, (name, prompt) in enumerate(STEPS, start=1):
        if idx not in wanted:
            continue
        print(f"   step {name}")
        res = call(MODEL_IMG, {
            "prompt": prompt,
            "image_url": image_url,
            "negative_prompt": NEGATIVE,
            "guidance_scale": 3.5,
            "num_images": 2,
            "output_format": "png",
            "safety_tolerance": "5",
        })
        urls = [i.get("url") for i in res.get("images", []) if i.get("url")]
        for n, u in enumerate(urls, start=1):
            dest = out / f"{name}_{n}.png"
            download(u, dest)
            results.append(dest)

    if not results:
        sys.exit("no images produced")

    best = results[0]
    print(f"[4/4] best = {best}")

    if args.video:
        print("   rendering video...")
        vurl = fal_client.upload_file(str(best))
        try:
            res = call(MODEL_VID, {
                "prompt": VIDEO_PROMPT,
                "image_url": vurl,
                "duration": "5",
                "aspect_ratio": "9:16",
                "negative_prompt": "blurry, deformed hands, extra fingers, morphing face",
            })
        except Exception as e:
            print(f"   kling failed ({e}), trying wan")
            res = call(MODEL_VID_FALLBACK, {
                "prompt": VIDEO_PROMPT,
                "image_url": vurl,
                "resolution": "720p",
            })
        u = first_url(res)
        if u:
            download(u, out / "05_video.mp4")

    print(f"\nDONE -> {out.resolve()}")


if __name__ == "__main__":
    main()