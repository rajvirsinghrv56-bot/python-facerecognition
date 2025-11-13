"""
Biometric Sentinel: Real-Time Verification System (Beginner-Friendly, Robust Loader)

Features:
- Loads "known faces" from the folder: known_faces/  (flat: Alice.jpg, Bob.png, etc.)
- Forces every image to 8-bit RGB (fixes 'Unsupported image type' errors).
- Opens webcam, recognizes faces live, draws labels.
- Logs first-seen-today to attendance.csv.
- Press 'q' to quit.

Install in your (face) conda env:
    pip install opencv-python face_recognition pandas pillow numpy
    # If dlib/face_recognition weren't installed:
    # conda install -n face -c conda-forge dlib face_recognition -y
"""

import cv2
import time
import pandas as pd
import face_recognition
from datetime import datetime
from pathlib import Path

# Robust image handling
from PIL import Image, UnidentifiedImageError
import numpy as np

# -------------- Configuration --------------
KNOWN_FACES_DIR = Path("known_faces")   # folder containing known face photos
ATTENDANCE_CSV = Path("attendance.csv") # where we log presence (append-only)
FRAME_DOWNSCALE = 0.25                  # speed-up by resizing frame (0.25 = 25% size)
MATCH_TOLERANCE = 0.5                   # 0.4–0.6 typical; lower is stricter
CAMERA_INDEX = 0                        # default webcam
WINDOW_TITLE = "Biometric Sentinel (press 'q' to quit)"
# -------------------------------------------

def safe_load_rgb_uint8(path: Path):
    """
    Load image safely as 8-bit RGB, C-contiguous numpy array.
    Falls back to OpenCV if PIL path fails.
    Returns np.ndarray or None.
    """
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            arr = np.asarray(im, dtype=np.uint8)
            if arr.ndim != 3 or arr.shape[2] != 3:
                print(f"[!] {path.name}: unexpected shape {arr.shape}, skipping.")
                return None
            # ensure contiguous memory
            if not arr.flags["C_CONTIGUOUS"]:
                arr = np.ascontiguousarray(arr)
            return arr
    except UnidentifiedImageError:
        print(f"[!] {path.name}: not an image or unsupported format (PIL). Trying OpenCV...")
    except Exception as e:
        print(f"[!] {path.name}: PIL load failed: {e}. Trying OpenCV...")

    # Fallback: OpenCV
    try:
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            print(f"[!] {path.name}: OpenCV couldn't read, skipping.")
            return None
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        if not rgb.flags["C_CONTIGUOUS"]:
            rgb = np.ascontiguousarray(rgb)
        return rgb
    except Exception as e:
        print(f"[!] {path.name}: OpenCV fallback failed: {e}")
        return None

def load_known_faces(known_dir: Path):
    """Load images from known_dir, convert to 8-bit RGB, compute encodings, and collect names."""
    if not known_dir.exists():
        known_dir.mkdir(parents=True, exist_ok=True)
        print(f"[i] Created {known_dir.resolve()} — add JPG/PNG face photos (Alice.jpg, Bob.png).")

    exts = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")
    files = [p for p in known_dir.iterdir() if p.is_file() and p.suffix.lower() in exts]

    if not files:
        print(f"[!] No images found in {known_dir}. Add clear, frontal face photos and rerun.")
        return [], []

    print("[i] Scanning known_faces:")
    for f in files:
        print("   -", f.name)

    known_encodings, known_names = [], []

    for img_path in files:
        name = img_path.stem  # file name without extension
        image = safe_load_rgb_uint8(img_path)
        if image is None:
            continue

        # Extra safety: enforce dtype/shape again
        if image.dtype != np.uint8:
            image = image.astype(np.uint8, copy=False)
        if image.ndim == 2:
            image = np.dstack([image, image, image])
        elif image.ndim == 3 and image.shape[2] != 3:
            print(f"[!] {img_path.name}: channel count {image.shape[2]} != 3, skipping.")
            continue
        if not image.flags["C_CONTIGUOUS"]:
            image = np.ascontiguousarray(image)

        try:
            encs = face_recognition.face_encodings(image)
        except Exception as e:
            print(f"[!] {img_path.name}: face_encodings error: {e}. Skipping.")
            continue

        if len(encs) == 0:
            print(f"[!] No face found in: {img_path.name} — use a clearer, frontal photo.")
            continue

        known_encodings.append(encs[0])
        known_names.append(name)
        print(f"[+] Loaded known face: {name} from {img_path.name}")

    return known_encodings, known_names

def init_attendance(csv_path: Path):
    """Create CSV if missing with columns: name, first_seen, date."""
    if not csv_path.exists():
        df = pd.DataFrame(columns=["name", "first_seen", "date"])
        df.to_csv(csv_path, index=False)
        print(f"[i] Created {csv_path.resolve()}")

def has_attendance_today(csv_path: Path, person_name: str) -> bool:
    """Return True if person_name already marked for today."""
    if not csv_path.exists():
        return False
    today = datetime.now().strftime("%Y-%m-%d")
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return False
    if df.empty:
        return False
    matched = df[(df["date"] == today) & (df["name"] == person_name)]
    return not matched.empty

def mark_attendance(csv_path: Path, person_name: str):
    """Append a row with current timestamp and date for the person's first sighting today."""
    now = datetime.now()
    row = {
        "name": person_name,
        "first_seen": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
    }
    header_needed = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        if header_needed:
            f.write("name,first_seen,date\n")
        f.write(f"{row['name']},{row['first_seen']},{row['date']}\n")
    print(f"[✓] Marked attendance for {person_name} at {row['first_seen']} on {row['date']}")

def main():
    # Prep
    known_encodings, known_names = load_known_faces(KNOWN_FACES_DIR)
    init_attendance(ATTENDANCE_CSV)

    if len(known_encodings) == 0:
        print("[!] No usable known faces found. Add clear face images to 'known_faces' and rerun.")
        return

    print("[i] Starting webcam... Press 'q' to quit.")
    video = cv2.VideoCapture(CAMERA_INDEX)

    if not video.isOpened():
        print("[!] Could not open webcam. Check CAMERA_INDEX or permissions.")
        return

    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_NORMAL)

    last_fps_time = time.time()
    frame_count = 0

    try:
        while True:
            ret, frame = video.read()
            if not ret:
                print("[!] Failed to read from webcam.")
                break

            small_frame = cv2.resize(frame, (0, 0), fx=FRAME_DOWNSCALE, fy=FRAME_DOWNSCALE)
            rgb_small = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)  # contiguous RGB
            rgb_small.setflags(write=1)  # ensure writeable (rarely needed)



            face_locations = face_recognition.face_locations(rgb_small, model="hog")
            face_encodings = face_recognition.face_encodings(rgb_small, face_locations)

            names_in_frame = []

            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=MATCH_TOLERANCE)
                name = "Unknown"

                if True in matches:
                    face_distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_index = face_distances.argmin()
                    if matches[best_index]:
                        name = known_names[best_index]

                names_in_frame.append(name)

                if name != "Unknown" and not has_attendance_today(ATTENDANCE_CSV, name):
                    mark_attendance(ATTENDANCE_CSV, name)

            for (top, right, bottom, left), name in zip(face_locations, names_in_frame):
                top = int(top / FRAME_DOWNSCALE)
                right = int(right / FRAME_DOWNSCALE)
                bottom = int(bottom / FRAME_DOWNSCALE)
                left = int(left / FRAME_DOWNSCALE)

                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.rectangle(frame, (left, bottom - 20), (right, bottom), (0, 255, 0), cv2.FILLED)
                cv2.putText(frame, name, (left + 6, bottom - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

            frame_count += 1
            now = time.time()
            if now - last_fps_time >= 1.0:
                fps = frame_count / (now - last_fps_time)
                last_fps_time = now
                frame_count = 0
                try:
                    cv2.setWindowTitle(WINDOW_TITLE, f"{WINDOW_TITLE}  |  FPS: {fps:.1f}")
                except Exception:
                    pass

            cv2.imshow(WINDOW_TITLE, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        video.release()
        cv2.destroyAllWindows()
        print("[i] Webcam released. Goodbye!]")

if __name__ == "__main__":
    main()
