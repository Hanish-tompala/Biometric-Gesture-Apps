"""
Smile-Selfie App (standalone)

Usage:
  - Install dependencies:
      pip install mediapipe opencv-python numpy
  - Run in PowerShell:
      python selfie_app.py
  - Smile at the camera! The app shows a 2-second countdown,
    takes a selfie, saves it with a timestamp, applies a filter,
    and displays the filtered image.

"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os
from datetime import datetime
import platform

# Try to import winsound for a simple beep on Windows
try:
    if platform.system().lower().startswith('win'):
        import winsound
    else:
        winsound = None
except Exception:
    winsound = None

# --- Configuration ---
SAVE_DIR = os.path.abspath(os.path.dirname(__file__))
FILTER = 'sepia'  # 'grayscale', 'sepia', 'cartoon', or None
DELAY_SECONDS = 2
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
MAX_FACES = 1
MIN_DETECTION_CONFIDENCE = 0.5
MIN_TRACKING_CONFIDENCE = 0.5
FRAME_SKIP = 0  # process every (FRAME_SKIP+1) frames; 0 -> process every frame
COOLDOWN_AFTER_CAPTURE = 2.0  # seconds to wait after a capture

# --- NEW: Smile Detection Threshold ---
# Empirical value. Resting face is ~0.2-0.3. A smile is > 0.35.
# Adjust this if it's too sensitive or not sensitive enough.
SMILE_THRESHOLD = 0.35

# Mediapipe helper
mp_face_mesh = mp.solutions.face_mesh

# --- Utility functions ---

def timestamp_filename(basename='selfie', ext='jpg'):
    t = datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{basename}_{t}.{ext}"


def apply_filter(img, mode='sepia'):
    """Apply a simple filter to a BGR OpenCV image and return BGR image."""
    if mode == 'grayscale':
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    elif mode == 'sepia':
        kernel = np.array([[0.272, 0.534, 0.131],
                           [0.349, 0.686, 0.168],
                           [0.393, 0.769, 0.189]])
        img_sepia = cv2.transform(img, kernel)
        img_sepia = np.clip(img_sepia, 0, 255).astype(np.uint8)
        return img_sepia
    elif mode == 'cartoon':
        img_color = cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.medianBlur(img_gray, 7)
        edges = cv2.adaptiveThreshold(img_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                      cv2.THRESH_BINARY, 9, 2)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        cartoon = cv2.bitwise_and(img_color, edges_colored)
        return cartoon
    else:
        return img

# --- NEW: Helper function for smile detection ---
def get_euclidean_dist(landmark1, landmark2):
    """Calculate 2D euclidean distance between two landmarks"""
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

# --- NEW: Smile Detection Logic ---
def detect_smile(landmarks):
    """
    Return True if a smile is detected based on landmark ratios.
    
    We measure the distance between lip corners (61, 291) and
    normalize it by the distance between cheek landmarks (234, 454).
    """
    try:
        # Lip corners
        left_corner = landmarks.landmark[61]
        right_corner = landmarks.landmark[291]
        
        # Face "width" reference points (cheeks)
        left_cheek = landmarks.landmark[234]
        right_cheek = landmarks.landmark[454]

        # Calculate distances
        mouth_width = get_euclidean_dist(left_corner, right_corner)
        face_width = get_euclidean_dist(left_cheek, right_cheek)

        if face_width < 1e-6: # avoid division by zero
            return False

        # Calculate ratio
        smile_ratio = mouth_width / face_width
        
        # Uncomment for debugging:
        # print(f"Smile Ratio: {smile_ratio:.4f}") 
        
        return smile_ratio > SMILE_THRESHOLD
    except Exception:
        return False

# --- NEW: Bounding Box Drawing ---
def draw_face_bbox(frame, landmarks):
    """Draw a bounding box around the detected face."""
    h, w, _ = frame.shape
    
    # Find min/max x and y from all 468 landmarks
    xmin = w
    ymin = h
    xmax = 0
    ymax = 0
    for landmark in landmarks.landmark:
        x = int(landmark.x * w)
        y = int(landmark.y * h)
        if x < xmin: xmin = x
        if x > xmax: xmax = x
        if y < ymin: ymin = y
        if y > ymax: ymax = y
    
    # Add some padding
    padding = 10
    xmin = max(0, xmin - padding)
    ymin = max(0, ymin - padding)
    xmax = min(w, xmax + padding)
    ymax = min(h, ymax + padding)

    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)


def play_beep():
    try:
        if winsound:
            # frequency, duration
            winsound.Beep(1000, 150)
        else:
            # fallback: print
            print('\a', end='', flush=True)
    except Exception:
        print('Beep fallback')


# --- Main app ---

def main():
    os.makedirs(SAVE_DIR, exist_ok=True)

    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)

    if not cap.isOpened():
        print('ERROR: Could not open camera. Make sure another app is not using it.')
        return

    # --- MODIFIED: Initialize Face Mesh ---
    face_mesh = mp_face_mesh.FaceMesh(
        max_num_faces=MAX_FACES,
        refine_landmarks=True,  # Get more accurate landmarks for lips/eyes
        min_detection_confidence=MIN_DETECTION_CONFIDENCE,
        min_tracking_confidence=MIN_TRACKING_CONFIDENCE)

    gesture_active = False
    gesture_start = 0
    captured = False
    last_capture_time = 0
    frame_idx = 0

    print('Press q to quit. Smile at the camera to trigger a selfie.')

    try:
        while True:
            # Read to 'frame_original'
            ret, frame_original = cap.read()
            if not ret:
                print('Failed to grab frame')
                break

            # Create flipped 'frame' for display only
            frame = cv2.flip(frame_original, 1)
            display_frame = frame.copy()

            # Optional: skip frames to improve speed
            if FRAME_SKIP > 0 and (frame_idx % (FRAME_SKIP + 1)) != 0:
                frame_idx += 1
                cv2.imshow('Live - Smile to take selfie', display_frame) # <-- Title Changed
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            # Convert for Mediapipe
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # --- MODIFIED: Process with Face Mesh ---
            results = face_mesh.process(rgb)

            h, w, _ = frame.shape

            # --- MODIFIED: Check for face landmarks ---
            if results.multi_face_landmarks:
                # We only care about the first face
                face_landmarks = results.multi_face_landmarks[0]
                
                # --- MODIFIED: Draw bounding box ---
                draw_face_bbox(display_frame, face_landmarks)

                # --- MODIFIED: Check for smile ---
                if detect_smile(face_landmarks):
                    # avoid repeated captures: check cooldown
                    if time.time() - last_capture_time < COOLDOWN_AFTER_CAPTURE:
                        # cooldown active
                        cv2.putText(display_frame, 'Cooldown...', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,255), 2)
                    else:
                        if not gesture_active:
                            gesture_active = True
                            gesture_start = time.time()
                            captured = False

                        elapsed = time.time() - gesture_start
                        remaining = max(0, int(DELAY_SECONDS - elapsed) + 1)
                        text = f'Taking Selfie in {max(0, round(DELAY_SECONDS - elapsed,1))}s'
                        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)
                        cv2.putText(display_frame, str(remaining), (w-120, 80), cv2.FONT_HERSHEY_DUPLEX, 3.0, (0,0,255), 6)

                        if elapsed >= DELAY_SECONDS and not captured:
                            # save selfie
                            filename = timestamp_filename('selfie', 'jpg')
                            full_path = os.path.join(SAVE_DIR, filename)
                            
                            # Save 'frame_original'
                            cv2.imwrite(full_path, frame_original)
                            print(f'Saved selfie to: {full_path}')

                            # apply filter
                            filtered = apply_filter(frame_original, mode=FILTER)
                            out_name = os.path.join(SAVE_DIR, 'filtered_' + filename)
                            cv2.imwrite(out_name, filtered)
                            print(f'Saved filtered selfie to: {out_name}')

                            # beep
                            play_beep()

                            # show filtered in a separate resizable window
                            try:
                                cv2.namedWindow('Filtered Selfie', cv2.WINDOW_NORMAL)
                                cv2.imshow('Filtered Selfie', filtered)
                            except Exception:
                                pass

                            captured = True
                            gesture_active = False
                            last_capture_time = time.time()
                else:
                    gesture_active = False
            else:
                gesture_active = False

            cv2.imshow('Live - Smile to take selfie', display_frame) # <-- Title Changed
            frame_idx += 1

            # quit on 'q' key
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print('Interrupted by user')
    finally:
        # --- MODIFIED: Close Face Mesh ---
        face_mesh.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()