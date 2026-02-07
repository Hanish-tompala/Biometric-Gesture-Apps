# Biometric-Gesture-Apps

A collection of interactive Python applications utilizing **OpenCV** and **MediaPipe** to perform real-time biometric tracking and gesture-based automation. These tools demonstrate the implementation of 2D/3D landmark normalization, facial expression analysis, and custom image processing filters.

---

## 🚀 Overview

This repository contains three primary applications:

### 1. 🔐 GestureLock: Biometric Unlock
A security prototype that uses hand geometry as a digital key.
* **Biometric Vector**: Converts 21 3D hand landmarks into a scale-invariant feature vector.
* **Normalization**: Landmarks are centered around the wrist and scaled by the "palm size" (distance from wrist to middle knuckle) to ensure the lock works regardless of distance from the camera.
* **Verification**: Uses Mean Absolute Error (MAE) to compare live gestures against the saved `.npy` key.

### 2. 😊 Smile-Selfie App
A hands-free camera that captures photos when the user smiles.
* [cite_start]**Landmark Ratio**: Detects smiles by calculating the ratio between lip corners and face width.
* [cite_start]**Face Mesh**: Employs 468-point face mesh tracking for high precision.
* [cite_start]**Auto-Capture**: Features a 2-second visual countdown and a 2.0s cooldown period to prevent duplicate captures.

### 3. ✌️ Peace-Sign Selfie App
An alternative selfie tool triggered by the "V" or Peace gesture.
* **Separation Logic**: Implements a horizontal separation check between index and middle fingers to reduce false positives.
* **Filter Suite**: Automatically applies filters such as **Sepia**, **Grayscale**, or **Cartoon** (edge-preserving blur + adaptive thresholding) to the final output.
* **Mirror Handling**: Provides a mirrored live preview for user comfort while saving the original, un-flipped frame to disk.

---

## 🛠️ Technical Implementation

| Feature | Methodology |
| :--- | :--- |
| **Face Tracking** | [cite_start]MediaPipe FaceMesh (Refine Landmarks = True)  |
| **Hand Tracking** | MediaPipe Hands (Model Complexity 0) |
| **Data Storage** | NumPy Binary Format (`.npy`) |
| **Image Filters** | 3x3 Transformation Matrices & Adaptive Thresholding |

---

## 📦 Installation & Usage

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/your-username/Biometric-Gesture-Apps.git](https://github.com/your-username/Biometric-Gesture-Apps.git)
   cd Biometric-Gesture-Apps

2. **Install dependencies**:

```bash
pip install opencv-python numpy mediapipe

