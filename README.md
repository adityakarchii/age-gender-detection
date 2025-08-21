# Age Detection Project

A simple age detection system using OpenCV. I built this to experiment with computer vision and see if I could guess people's ages from their faces.

# Age Detection Project

A simple age detection system using OpenCV. I built this to experiment with computer vision and see if I could guess people's ages from their faces.

# Age & Gender Detection

Real-time age and gender detection using OpenCV and computer vision.

## Features

- Live webcam face detection
- Age estimation with categories
- Gender prediction (Male/Female)
- Confidence scoring
- Real-time processing

## Files

- `simple_age_gender.py` - Basic version with color-coded results
- `age_gender_detector.py` - Advanced version with confidence levels

## Setup

```bash
pip install opencv-python numpy
```

## Usage

Basic version:
```bash
python simple_age_gender.py
```

Advanced version:
```bash
python age_gender_detector.py
```

## Controls

- `q` - Quit
- `s` - Save screenshot (advanced version only)

## Output

- Blue boxes for males, pink for females (basic)
- Green/Yellow/Orange boxes based on confidence (advanced)
- Age categories and estimated years displayed
- Real-time confidence percentages
