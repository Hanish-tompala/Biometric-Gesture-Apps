# Biometric Gesture Apps

## Overview
This repository contains three innovative applications that utilize biometric gestures to enhance user interaction:

1. **GestureLock**: A secure method of unlocking devices using unique gesture patterns.
2. **Smile-Selfie App**: Automatically captures photos when the user smiles, using real-time facial recognition.
3. **Peace-Sign Selfie App**: Captures photos when the user displays a peace sign, incorporating hand gesture recognition.

## Installation Instructions
To install the applications, follow these steps:
. Clone the repository to your local machine:
   ```bash
   git clone https://github.com/Hanish-tompala/Biometric-Gesture-Apps.git
   cd Biometric-Gesture-Apps
   ```

## Dependencies
These applications require the following libraries:
- OpenCV
- MediaPipe
- NumPy

## Features
- **GestureLock**: 
  - Customizable gesture patterns for secure unlocking.
  - User-friendly interface for setting up gestures.
- **Smile-Selfie App**: 
  - Real-time detection and capturing based on smile recognition.
  - Adjustable sensitivity for smile detection.
- **Peace-Sign Selfie App**: 
  - Captures images when detecting a peace sign.
  - Supports multiple modes (single/timed capture).

## Usage Instructions
### Keyboard Controls:
- **GestureLock**: 
  - Use arrow keys to navigate through options.
  - Press 'Enter' to select or confirm a gesture.
- **Smile-Selfie App**: 
  - Press 'S' to start/stop the camera.
  - 'C' to toggle smile detection sensitivity.
- **Peace-Sign Selfie App**: 
  - Press 'P' to initiate the camera for peace sign detection.

### Configuration Options
Each application allows configuration through a settings file. You can customize:
- Sensitivity levels for detection.
- File paths for saved images.

## Technical Implementation Details
- **Face Tracking**: Utilizes MediaPipe for real-time face detection.
- **Hand Tracking**: Implements hand tracking via MediaPipe to identify gestures.
- **Biometric Algorithms**: Integrated algorithms for gesture recognition that learn and adapt over time.
- **Filter Implementations**: Various filters are applied to enhance image quality in captured photographs.

## Requirements
- Python 3.x
- A working webcam
- A compatible operating system (Windows/Linux/Mac)

**Last updated: 2026-02-07 06:01:54 (UTC)**
