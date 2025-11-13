# Normalizes every image in known_faces/ to plain 8-bit RGB JPEG (no alpha/CMYK),
# applies EXIF orientation, and saves as *_norm.jpg next to the original.

from pathlib import Path
from PIL import Image, ImageOps, UnidentifiedImageError

KNOWN = Path("known_faces")
OUT   = KNOWN  # write normalized images alongside originals

exts = {".jpg", ".jpeg", ".png", ".bmp", ".jfif", ".webp", ".tif", ".tiff"}

def normalize_one(p: Path):
    try:
        with Image.open(p) as im:
            # fix EXIF orientation, force RGB 8-bit
            im = ImageOps.exif_transpose(im).convert("RGB")
            out = OUT / f"{p.stem}_norm.jpg"
            im.save(out, format="JPEG", quality=95, subsampling="4:2:0", optimize=True)
            print(f"[✓] Saved {out.name}")
            return True
    except UnidentifiedImageError:
        print(f"[!] Not an image / unsupported: {p.name}")
    except Exception as e:
        print(f"[!] Failed {p.name}: {e}")
    return False

def main():
    if not KNOWN.exists():
        print(f"[!] Missing {KNOWN}/ — create it and add your photos first.")
        return
    files = [p for p in KNOWN.iterdir() if p.is_file() and p.suffix.lower() in exts]
    if not files:
        print(f"[!] No images found in {KNOWN}/")
        return
    for p in files:
        normalize_one(p)

if __name__ == "__main__":
    main()
