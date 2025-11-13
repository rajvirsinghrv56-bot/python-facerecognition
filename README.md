
# Biometric Sentinel (Beginner-Friendly)

A very simple, real-time face recognition presence tracker using:
- OpenCV (camera + drawing)
- face_recognition (encodings & matching)
- pandas (CSV attendance log)

## Quick Start
1) Create a Python environment (conda or venv) and install deps:
   ```bash
   pip install -r requirements.txt
   ```

2) Add some known faces (images with a single clear face) into `known_faces/`
   Example:
   ```text
   known_faces/
     Alice.jpg
     Bob.png
   ```

3) Run the app:
   ```bash
   python biometric_sentinel.py
   ```

4) Press `q` to quit. Check `attendance.csv` for logs.

## Tips
- If recognition is poor, add 2–3 images per person in different lighting/angles.
- Adjust `MATCH_TOLERANCE` in the script (0.4 is stricter, 0.6 is looser).
- If `dlib` fails to install, try conda-forge or prebuilt wheels.
