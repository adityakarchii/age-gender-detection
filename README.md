# Age & Gender Detection System

Real-time age and gender detection using OpenCV and computer vision techniques.

## Features

- **Live webcam face detection**
- **Age estimation** with detailed categories
- **Gender classification** (Male/Female)
- **Confidence scoring** for reliability
- **Real-time processing** at 30+ FPS
- **Session statistics** and analytics

## Files

- `age_gender_detector.py` - Advanced version with comprehensive features
- `simple_age_gender.py` - Streamlined version with essential functionality
- `requirements.txt` - Project dependencies
- `README.md` - This documentation

## Quick Setup

```bash
# Install dependencies
pip install opencv-python numpy

# Run the advanced version
python age_gender_detector.py

# Or run the simple version
python simple_age_gender.py
```

## Controls

**Advanced Version:**
- `q` - Quit application
- `s` - Save screenshot
- `r` - Reset statistics
- `t` - Toggle tracking
- `e` - Export session data
- `h` - Hide/show overlay

**Simple Version:**
- `q` - Quit application
- `s` - Toggle statistics panel
- `e` - Toggle enhanced processing
- `c` - Capture screenshot

## How It Works

The system analyzes facial features including:
- Jawline definition and facial structure
- Skin texture and wrinkle patterns
- Eye region characteristics
- Overall facial proportions

Age detection considers multiple factors like face area, wrinkle density, and regional aging indicators to categorize faces into age groups from babies to seniors.

Gender classification uses geometric analysis of facial features, contrast patterns, and structural characteristics to determine male/female classification with confidence scoring.

## Requirements

- Python 3.7+
- Webcam/Camera
- Good lighting conditions for best accuracy

## Output

- **Color-coded detection boxes** based on confidence levels
- **Age categories** with estimated years
- **Gender classification** with confidence percentages
- **Real-time statistics** and session analytics

Built with OpenCV for computer vision and NumPy for numerical processing.