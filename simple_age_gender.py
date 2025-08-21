import cv2
import numpy as np
import time
import math
import os
from collections import defaultdict

class SimpleDetectionStats:
    def __init__(self):
        self.face_count = 0
        self.gender_count = {'Male': 0, 'Female': 0}
        self.age_count = defaultdict(int)
        self.detection_history = []
        self.start_time = time.time()
    
    def update(self, gender, age_category):
        self.face_count += 1
        self.gender_count[gender] += 1
        self.age_count[age_category] += 1
        self.detection_history.append({
            'gender': gender,
            'age': age_category,
            'timestamp': time.time()
        })
    
    def get_session_duration(self):
        return int(time.time() - self.start_time)
    
    def get_dominant_gender(self):
        return max(self.gender_count.items(), key=lambda x: x[1])[0] if self.face_count > 0 else "Unknown"
    
    def get_dominant_age(self):
        return max(self.age_count.items(), key=lambda x: x[1])[0] if self.age_count else "Unknown"

def enhance_image_quality(img):
    """Apply basic image enhancement"""
    # Convert to LAB color space for better processing
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)
    
    # Merge channels back
    enhanced_lab = cv2.merge([l, a, b])
    enhanced_img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    return enhanced_img

def calculate_face_symmetry(gray_face):
    """Calculate facial symmetry score"""
    height, width = gray_face.shape
    
    # Split face into left and right halves
    left_half = gray_face[:, :width//2]
    right_half = gray_face[:, width//2:]
    
    # Flip right half to compare with left
    right_half_flipped = cv2.flip(right_half, 1)
    
    # Resize to match if needed
    min_width = min(left_half.shape[1], right_half_flipped.shape[1])
    left_half = left_half[:, :min_width]
    right_half_flipped = right_half_flipped[:, :min_width]
    
    # Calculate difference
    if left_half.shape == right_half_flipped.shape:
        difference = cv2.absdiff(left_half, right_half_flipped)
        symmetry_score = 1.0 - (np.mean(difference) / 255.0)
        return max(0, symmetry_score)
    
    return 0.5  # Default symmetry score

def detect_facial_features(gray_face):
    """Detect and analyze facial features"""
    height, width = gray_face.shape
    
    # Eye region analysis (upper third)
    eye_region = gray_face[int(height*0.15):int(height*0.45), int(width*0.15):int(width*0.85)]
    eye_brightness = np.mean(eye_region) if eye_region.size > 0 else 128
    
    # Nose region analysis (middle)
    nose_region = gray_face[int(height*0.35):int(height*0.65), int(width*0.35):int(width*0.65)]
    nose_contrast = np.std(nose_region) if nose_region.size > 0 else 0
    
    # Mouth region analysis (lower third)
    mouth_region = gray_face[int(height*0.65):int(height*0.9), int(width*0.2):int(width*0.8)]
    mouth_activity = cv2.Laplacian(mouth_region, cv2.CV_64F).var() if mouth_region.size > 0 else 0
    
    return {
        'eye_brightness': eye_brightness,
        'nose_contrast': nose_contrast,
        'mouth_activity': mouth_activity
    }

def guess_gender(face_img, confidence_mode=True):
    # Enhanced image processing
    enhanced_face = enhance_image_quality(face_img)
    gray = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    aspect_ratio = width / height
    
    # Calculate facial symmetry
    symmetry_score = calculate_face_symmetry(gray)
    
    # Detect facial features
    features = detect_facial_features(gray)
    
    # Enhanced jawline analysis
    jaw_area = gray[int(height*0.65):, :]
    jaw_edges = cv2.Canny(jaw_area, 40, 120)
    jaw_strength = np.sum(jaw_edges) / max(jaw_edges.size, 1)
    
    # Cheekbone analysis
    cheek_region = gray[int(height*0.3):int(height*0.7), :]
    cheek_definition = cv2.Laplacian(cheek_region, cv2.CV_64F).var()
    
    # Overall facial contrast and texture
    overall_contrast = np.std(gray)
    texture_roughness = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Forehead analysis
    forehead_region = gray[0:int(height*0.35), int(width*0.2):int(width*0.8)]
    forehead_smoothness = 255 - np.std(forehead_region) if forehead_region.size > 0 else 128
    
    # Advanced scoring system
    male_score = 0
    female_score = 0
    
    # Aspect ratio scoring (enhanced)
    if aspect_ratio > 0.85:
        male_score += 2.5
    elif aspect_ratio > 0.82:
        male_score += 1.5
    elif aspect_ratio < 0.78:
        female_score += 2.0
    else:
        female_score += 1.0
    
    # Jawline strength (primary indicator)
    if jaw_strength > 15:
        male_score += 3.0
    elif jaw_strength > 12:
        male_score += 2.0
    elif jaw_strength < 8:
        female_score += 2.5
    else:
        female_score += 1.0
    
    # Cheekbone definition
    if cheek_definition > 100:
        male_score += 1.5
    else:
        female_score += 1.5
    
    # Overall contrast
    if overall_contrast > 25:
        male_score += 2.0
    elif overall_contrast > 22:
        male_score += 1.0
    elif overall_contrast < 18:
        female_score += 2.0
    
    # Texture analysis
    if texture_roughness > 80:
        male_score += 1.5
    else:
        female_score += 1.0
    
    # Forehead smoothness
    if forehead_smoothness > 200:
        female_score += 1.5
    else:
        male_score += 0.5
    
    # Symmetry factor (generally higher in females)
    if symmetry_score > 0.8:
        female_score += 1.0
    elif symmetry_score < 0.6:
        male_score += 0.5
    
    # Eye brightness factor
    if features['eye_brightness'] > 140:
        female_score += 1.0
    
    # Nose contrast
    if features['nose_contrast'] > 20:
        male_score += 1.0
    
    # Final decision with confidence
    total_score = male_score + female_score
    if total_score > 0:
        if male_score > female_score:
            confidence = male_score / total_score
            return "Male", confidence if confidence_mode else "Male"
        else:
            confidence = female_score / total_score
            return "Female", confidence if confidence_mode else "Female"
    
    return ("Unknown", 0.5) if confidence_mode else "Unknown"

def detect_age_simple(face_img, enhanced_mode=True):
    # Enhanced preprocessing
    if enhanced_mode:
        enhanced_face = enhance_image_quality(face_img)
        gray = cv2.cvtColor(enhanced_face, cv2.COLOR_BGR2GRAY)
    else:
        gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    brightness = np.mean(gray)
    face_area = height * width
    
    # Multi-scale edge detection for better wrinkle analysis
    edges_fine = cv2.Canny(gray, 20, 80)    # Fine details
    edges_coarse = cv2.Canny(gray, 40, 120)  # Coarse features
    
    wrinkle_density_fine = np.sum(edges_fine > 0) / max(edges_fine.size, 1)
    wrinkle_density_coarse = np.sum(edges_coarse > 0) / max(edges_coarse.size, 1)
    
    # Skin texture analysis
    skin_texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    smoothness_factor = 1.0 / (1.0 + skin_texture / 100)
    
    # Eye region aging analysis
    eye_region = gray[int(height*0.15):int(height*0.45), int(width*0.15):int(width*0.85)]
    if eye_region.size > 0:
        eye_wrinkles = cv2.Canny(eye_region, 30, 100)
        eye_aging = np.sum(eye_wrinkles > 0) / eye_wrinkles.size
    else:
        eye_aging = 0
    
    # Forehead lines detection
    forehead_region = gray[0:int(height*0.3), int(width*0.2):int(width*0.8)]
    if forehead_region.size > 0:
        forehead_lines = cv2.Canny(forehead_region, 25, 90)
        forehead_aging = np.sum(forehead_lines > 0) / forehead_lines.size
    else:
        forehead_aging = 0
    
    # Combined aging score
    overall_aging = (wrinkle_density_coarse + eye_aging + forehead_aging) / 3
    
    # Enhanced age categorization with multiple factors
    if face_area < 8000 and brightness > 125 and overall_aging < 0.04 and smoothness_factor > 0.8:
        return "Baby", "~2 years", overall_aging
    elif face_area < 12000 and overall_aging < 0.06 and brightness > 115:
        return "Toddler", "~4 years", overall_aging
    elif face_area < 18000 and overall_aging < 0.08 and smoothness_factor > 0.6:
        return "Child", "~8 years", overall_aging
    elif overall_aging < 0.12 and face_area > 15000 and eye_aging < 0.10:
        return "Teen", "~16 years", overall_aging
    elif overall_aging < 0.18 and eye_aging < 0.15 and forehead_aging < 0.12:
        return "Young Adult", "~22 years", overall_aging
    elif overall_aging < 0.25 and forehead_aging < 0.20:
        return "Adult", "~30 years", overall_aging
    elif overall_aging < 0.35 and eye_aging < 0.30:
        return "Middle Age", "~45 years", overall_aging
    elif overall_aging < 0.45:
        return "Mature", "~60 years", overall_aging
    else:
        return "Senior", "~70 years", overall_aging

def create_color_gradient(base_color, confidence):
    """Create color gradient based on confidence"""
    if confidence > 0.8:
        return base_color
    elif confidence > 0.6:
        # Blend with yellow for medium confidence
        blend_factor = (confidence - 0.6) / 0.2
        return tuple(int(base_color[i] * blend_factor + (0, 255, 255)[i] * (1 - blend_factor)) for i in range(3))
    else:
        # Blend with red for low confidence
        blend_factor = confidence / 0.6
        return tuple(int(base_color[i] * blend_factor + (0, 0, 255)[i] * (1 - blend_factor)) for i in range(3))

def draw_enhanced_info_box(frame, x, y, w, h, gender, age_cat, age_est, confidence, aging_score, color):
    """Draw enhanced information box with better styling"""
    box_height = 100
    box_width = max(w, 300)
    
    # Draw semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y-box_height), (x+box_width, y), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)
    
    # Draw colored border
    cv2.rectangle(frame, (x-2, y-box_height-2), (x+box_width+2, y+2), color, 3)
    
    # Multi-line text with different colors and sizes
    cv2.putText(frame, f"{gender} - {age_cat}", (x+8, y-75), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    
    cv2.putText(frame, f"Estimated: {age_est}", (x+8, y-55), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Confidence: {confidence:.1%}", (x+8, y-35), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    
    cv2.putText(frame, f"Aging Score: {aging_score:.3f}", (x+8, y-15), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (180, 180, 180), 1)

def draw_statistics_panel(frame, stats):
    """Draw real-time statistics panel"""
    panel_width = 320
    panel_height = 180
    x_pos = frame.shape[1] - panel_width - 10
    y_pos = 10
    
    # Semi-transparent background
    overlay = frame.copy()
    cv2.rectangle(overlay, (x_pos, y_pos), (x_pos + panel_width, y_pos + panel_height), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    # Border
    cv2.rectangle(frame, (x_pos, y_pos), (x_pos + panel_width, y_pos + panel_height), (255, 255, 255), 2)
    
    # Title
    cv2.putText(frame, "Session Statistics", (x_pos + 10, y_pos + 25), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    # Statistics
    duration = stats.get_session_duration()
    cv2.putText(frame, f"Duration: {duration//60}m {duration%60}s", (x_pos + 10, y_pos + 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Total Faces: {stats.face_count}", (x_pos + 10, y_pos + 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)
    
    cv2.putText(frame, f"Male: {stats.gender_count['Male']} | Female: {stats.gender_count['Female']}", (x_pos + 10, y_pos + 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    if stats.face_count > 0:
        cv2.putText(frame, f"Most Common Age: {stats.get_dominant_age()}", (x_pos + 10, y_pos + 110), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        cv2.putText(frame, f"Dominant Gender: {stats.get_dominant_gender()}", (x_pos + 10, y_pos + 130), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
    
    # Detection rate
    if duration > 0:
        rate = stats.face_count / duration * 60  # per minute
        cv2.putText(frame, f"Rate: {rate:.1f} faces/min", (x_pos + 10, y_pos + 150), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

def main():
    # Initialize enhanced statistics and settings
    stats = SimpleDetectionStats()
    show_stats = True
    enhanced_processing = True
    save_detections = False
    
    # Enhanced camera setup
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    # Optimized camera settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Multiple cascade classifiers for better detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    profile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
    
    if face_cascade.empty():
        print("Error: Could not load face cascade!")
        return
    
    print("=== Simple Age & Gender Detection ===")
    print("Controls:")
    print("  'q' - Quit")
    print("  's' - Toggle statistics panel")
    print("  'e' - Toggle enhanced processing")
    print("  'd' - Toggle detection saving")
    print("  'r' - Reset statistics")
    print("  'c' - Capture screenshot")
    
    frame_count = 0
    fps_count = 0
    fps_start_time = time.time()
    current_fps = 0
    
    # Create output directory for screenshots
    output_dir = "simple_detection_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_count += 1
        
        # Calculate FPS every 30 frames
        if fps_count >= 30:
            current_time = time.time()
            current_fps = fps_count / (current_time - fps_start_time)
            fps_start_time = current_time
            fps_count = 0
        
        # Mirror effect
        frame = cv2.flip(frame, 1)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Enhanced face detection with multiple scales
        faces_frontal = face_cascade.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(80, 80),
            maxSize=(500, 500),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        # Optional: Add profile face detection for better coverage
        faces_profile = []
        if len(faces_frontal) == 0 and frame_count % 3 == 0:  # Every 3rd frame
            faces_profile = profile_cascade.detectMultiScale(gray, 1.2, 4, minSize=(60, 60))
        
        # Combine detections
        all_faces = list(faces_frontal) + list(faces_profile)
        
        # Process each detected face
        for (x, y, w, h) in all_faces:
            # Validate face size
            if w < 60 or h < 60:
                continue
                
            # Extract face region with padding
            padding = 10
            face_x1 = max(0, x - padding)
            face_y1 = max(0, y - padding)
            face_x2 = min(frame.shape[1], x + w + padding)
            face_y2 = min(frame.shape[0], y + h + padding)
            
            face = frame[face_y1:face_y2, face_x1:face_x2]
            
            if face.size > 0:
                try:
                    # Enhanced detection with confidence
                    age_cat, age_est, aging_score = detect_age_simple(face, enhanced_processing)
                    gender, gender_confidence = guess_gender(face, confidence_mode=True)
                    
                    # Update statistics
                    stats.update(gender, age_cat)
                    
                    # Determine base colors
                    if gender == "Male":
                        base_color = (255, 100, 100)  # Light blue
                    else:
                        base_color = (255, 100, 255)  # Light pink
                    
                    # Apply confidence-based color gradient
                    final_color = create_color_gradient(base_color, gender_confidence)
                    
                    # Draw enhanced bounding box
                    thickness = max(2, int(gender_confidence * 4))
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), final_color, thickness)
                    
                    # Draw enhanced info box
                    draw_enhanced_info_box(frame, x, y, w, h, gender, age_cat, age_est, 
                                         gender_confidence, aging_score, final_color)
                    
                    # Optional: Save detection data
                    if save_detections and frame_count % 60 == 0:  # Every 2 seconds at 30fps
                        detection_data = {
                            'timestamp': time.time(),
                            'gender': gender,
                            'age_category': age_cat,
                            'confidence': gender_confidence,
                            'aging_score': aging_score
                        }
                        # Could save to file here
                        
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        
        # Draw statistics panel
        if show_stats:
            draw_statistics_panel(frame, stats)
        
        # Draw FPS counter
        cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, frame.shape[0] - 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Draw processing mode indicator
        mode_text = "Enhanced" if enhanced_processing else "Basic"
        cv2.putText(frame, f"Mode: {mode_text}", (10, frame.shape[0] - 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Draw main title with enhanced styling
        title_text = "Simple Age & Gender Detection System"
        cv2.putText(frame, title_text, (10, 35), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Draw detection count
        cv2.putText(frame, f"Faces Detected: {stats.face_count}", (10, frame.shape[0] - 10), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Show frame
        cv2.imshow('Simple Age & Gender Detection', frame)
        
        # Enhanced keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            show_stats = not show_stats
            print(f"Statistics panel: {'ON' if show_stats else 'OFF'}")
        elif key == ord('e'):
            enhanced_processing = not enhanced_processing
            print(f"Enhanced processing: {'ON' if enhanced_processing else 'OFF'}")
        elif key == ord('d'):
            save_detections = not save_detections
            print(f"Detection saving: {'ON' if save_detections else 'OFF'}")
        elif key == ord('r'):
            stats = SimpleDetectionStats()
            print("Statistics reset!")
        elif key == ord('c'):
            timestamp = int(time.time())
            filename = os.path.join(output_dir, f"simple_detection_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"Screenshot saved: {filename}")
    
    # Final session summary
    duration = stats.get_session_duration()
    print(f"\n=== Session Summary ===")
    print(f"Total duration: {duration//60}m {duration%60}s")
    print(f"Total faces detected: {stats.face_count}")
    print(f"Male detections: {stats.gender_count['Male']}")
    print(f"Female detections: {stats.gender_count['Female']}")
    if stats.face_count > 0:
        print(f"Most common age group: {stats.get_dominant_age()}")
        print(f"Detection rate: {stats.face_count/max(1, duration)*60:.1f} faces/minute")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
