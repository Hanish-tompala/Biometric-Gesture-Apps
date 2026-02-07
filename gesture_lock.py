"""
GestureLock: A Biometric Unlock App

Usage:
  - Install dependencies:
      pip install opencv-python numpy mediapipe
  - Run the app:
      python gesture_lock.py

Instructions:
  - On first run (no 'gesture_key.npy' found):
    1. Make a unique, clear hand gesture in front of the camera.
    2. Press 'r' to register your gesture as the key.
  
  - On subsequent runs:
    1. The app will show "LOCKED".
    2. Show your registered gesture to the camera.
    3. The app will change to "UNLOCKED" for 3 seconds.

  - Other keys:
    'd' - Delete the saved key to re-register.
    'q' - Quit the application.
"""

import cv2
import numpy as np
import mediapipe as mp
import time
import os

# --- Configuration ---
KEY_FILE = "gesture_key.npy"
SIMILARITY_THRESHOLD = 0.08  # How close the gestures must be. Lower = stricter.
UNLOCK_DURATION = 3.0       # How long to stay unlocked (in seconds).

# --- MediaPipe Setup ---
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    model_complexity=0,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# --- Core Biometric Functions ---

def normalize_gesture(hand_landmarks):
    """
    Converts 21 3D hand landmarks into a scale-invariant and
    position-invariant feature vector.
    
    1.  Get all 21 (x, y, z) points.
    2.  Center them around the wrist (landmark 0).
    3.  Normalize them by the "palm size" (dist from wrist 0 to middle knuckle 9)
        to make the gesture scale-invariant.
    """
    landmarks_list = []
    for landmark in hand_landmarks.landmark:
        landmarks_list.append([landmark.x, landmark.y, landmark.z])
    
    # Convert to numpy array
    landmarks_np = np.array(landmarks_list)
    
    # 1. Center around the wrist
    base_point = landmarks_np[0]  # Wrist
    relative_landmarks = landmarks_np - base_point
    
    # 2. Calculate palm size (distance from wrist to middle finger knuckle)
    palm_vec = relative_landmarks[9] # Middle finger knuckle
    palm_size = np.linalg.norm(palm_vec)
    
    if palm_size < 1e-6:
        # Avoid division by zero if hand is not detected properly
        return None
        
    # 3. Normalize by palm size
    normalized_landmarks = relative_landmarks / palm_size
    
    # Return as a flat 1D vector (21*3 = 63 features)
    return normalized_landmarks.flatten()


def compare_gestures(gesture1_flat, gesture2_flat):
    """
    Compares two normalized gesture vectors and returns a similarity score.
    We use Mean Absolute Error (MAE).
    A lower score means the gestures are more similar.
    """
    if gesture1_flat is None or gesture2_flat is None:
        return float('inf')
        
    error = np.mean(np.abs(gesture1_flat - gesture2_flat))
    return error

# --- Main Application ---

def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    saved_key = None
    state = "ENROLL"
    
    # --- "Testing App" Logic: Load Key ---
    if os.path.exists(KEY_FILE):
        try:
            saved_key = np.load(KEY_FILE)
            state = "LOCKED"
            print(f"Key '{KEY_FILE}' loaded. App is LOCKED.")
        except Exception as e:
            print(f"Error loading key: {e}. Please register a new key.")
            state = "ENROLL"
    else:
        print("No key found. Please register your unlock gesture.")

    unlock_time = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame.")
                break
                
            frame = cv2.flip(frame, 1)
            display_frame = frame.copy()
            h, w, _ = frame.shape

            # Convert for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb_frame)

            current_gesture = None
            
            # --- Gesture Detection ---
            if results.multi_hand_landmarks:
                hand_landmarks = results.multi_hand_landmarks[0]
                mp_drawing.draw_landmarks(
                    display_frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)
                
                # Normalize the current gesture
                current_gesture = normalize_gesture(hand_landmarks)

            # --- State Machine Logic ---
            
            key = cv2.waitKey(1) & 0xFF

            if state == "ENROLL":
                cv2.putText(display_frame, "REGISTER GESTURE", (10, 40), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(display_frame, "Hold gesture and press 'r'", (10, 80), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                if key == ord('r') and current_gesture is not None:
                    # Save the key
                    np.save(KEY_FILE, current_gesture)
                    saved_key = current_gesture
                    state = "LOCKED"
                    print(f"Key saved to '{KEY_FILE}'! App is now LOCKED.")

            elif state == "LOCKED":
                # Draw "LOCKED" UI
                cv2.rectangle(display_frame, (w // 2 - 100, h // 2 - 40), 
                              (w // 2 + 100, h // 2 + 40), (0, 0, 255), -1)
                cv2.putText(display_frame, "LOCKED", (w // 2 - 75, h // 2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

                # --- Unlock Logic ---
                if current_gesture is not None:
                    similarity = compare_gestures(saved_key, current_gesture)
                    
                    # Uncomment to debug the similarity score
                    # cv2.putText(display_frame, f"Sim: {similarity:.4f}", (10, 120), 
                    #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

                    if similarity < SIMILARITY_THRESHOLD:
                        state = "UNLOCKED"
                        unlock_time = time.time()
                        print("UNLOCKED!")
            
            elif state == "UNLOCKED":
                # Draw "UNLOCKED" UI
                cv2.rectangle(display_frame, (w // 2 - 120, h // 2 - 40), 
                              (w // 2 + 120, h // 2 + 40), (0, 255, 0), -1)
                cv2.putText(display_frame, "UNLOCKED", (w // 2 - 110, h // 2 + 15), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 3)
                
                # Check if it's time to re-lock
                if time.time() - unlock_time > UNLOCK_DURATION:
                    state = "LOCKED"
                    print("Re-locking...")

            # --- Handle 'd' (delete) and 'q' (quit) ---
            if key == ord('q'):
                print("Quitting.")
                break
                
            if key == ord('d'):
                if os.path.exists(KEY_FILE):
                    os.remove(KEY_FILE)
                    print(f"Key '{KEY_FILE}' deleted.")
                saved_key = None
                state = "ENROLL"
                print("Please register a new key.")

            cv2.imshow("GestureLock Biometric App", display_frame)

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        hands.close()
        cap.release()
        cv2.destroyAllWindows()
        print("App closed.")

if __name__ == "__main__":
    main()
