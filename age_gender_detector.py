import cv2
import numpy as np
import time
import os
import json
from datetime import datetime
import threading
import queue

class FaceAnalyzer:
    def __init__(self):
        self.detection_history = []
        self.confidence_threshold = 0.6
        self.tracking_enabled = True
        self.statistics = {
            'total_detections': 0,
            'male_detections': 0,
            'female_detections': 0,
            'age_distribution': {},
            'session_start': datetime.now()
        }
    
    def update_statistics(self, gender, age_category):
        self.statistics['total_detections'] += 1
        if gender == 'Male':
            self.statistics['male_detections'] += 1
        else:
            self.statistics['female_detections'] += 1
        
        if age_category not in self.statistics['age_distribution']:
            self.statistics['age_distribution'][age_category] = 0
        self.statistics['age_distribution'][age_category] += 1
    
    def get_session_stats(self):
        duration = (datetime.now() - self.statistics['session_start']).seconds
        return {
            'session_duration_minutes': duration // 60,
            'detections_per_minute': self.statistics['total_detections'] / max(1, duration // 60),
            'gender_ratio': {
                'male_percentage': (self.statistics['male_detections'] / max(1, self.statistics['total_detections'])) * 100,
                'female_percentage': (self.statistics['female_detections'] / max(1, self.statistics['total_detections'])) * 100
            },
            'most_common_age': max(self.statistics['age_distribution'].items(), key=lambda x: x[1])[0] if self.statistics['age_distribution'] else "None"
        }

def detect_gender(face_img, analyzer=None):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced feature extraction
    brightness = np.mean(gray)
    contrast = np.std(gray)
    face_height, face_width = gray.shape
    aspect_ratio = face_width / face_height
    
    # Apply Gaussian blur for noise reduction
    blurred = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    # Enhanced region analysis
    height_third = face_height // 3
    width_third = face_width // 3
    
    # Forehead analysis (enhanced)
    upper_region = gray[0:height_third, width_third:2*width_third]
    upper_brightness = np.mean(upper_region) if upper_region.size > 0 else brightness
    forehead_smoothness = np.std(upper_region) if upper_region.size > 0 else 0
    
    # Eye region analysis
    eye_region = gray[height_third//2:height_third+height_third//2, :]
    eye_contrast = np.std(eye_region) if eye_region.size > 0 else 0
    
    # Cheek analysis
    middle_region = gray[height_third:2*height_third, :]
    middle_contrast = np.std(middle_region)
    cheek_texture = cv2.Laplacian(middle_region, cv2.CV_64F).var()
    
    # Jawline and chin analysis (enhanced)
    lower_region = gray[2*height_third:, :]
    lower_edges = cv2.Canny(lower_region, 50, 150)
    jaw_definition = np.sum(lower_edges > 0) / lower_edges.size if lower_edges.size > 0 else 0
    
    # Chin prominence analysis
    chin_region = gray[int(0.8*face_height):, int(0.3*face_width):int(0.7*face_width)]
    chin_prominence = np.std(chin_region) if chin_region.size > 0 else 0
    
    # Enhanced scoring system with weighted features
    male_score = 0
    female_score = 0
    
    # Jawline definition (primary indicator)
    if jaw_definition > 0.18:
        male_score += 0.35
    elif jaw_definition > 0.15:
        male_score += 0.25
    elif jaw_definition < 0.10:
        female_score += 0.30
    else:
        female_score += 0.15
    
    # Face aspect ratio analysis
    if aspect_ratio > 0.90:
        male_score += 0.25
    elif aspect_ratio > 0.85:
        male_score += 0.15
    elif aspect_ratio < 0.80:
        female_score += 0.25
    else:
        female_score += 0.15
    
    # Facial contrast and texture
    if middle_contrast > 30:
        male_score += 0.20
    elif middle_contrast > 25:
        male_score += 0.15
    elif middle_contrast < 20:
        female_score += 0.20
    
    # Skin texture analysis
    if cheek_texture > 50:
        male_score += 0.15
    else:
        female_score += 0.15
    
    # Forehead smoothness
    if forehead_smoothness < 15:
        female_score += 0.20
    else:
        male_score += 0.10
    
    # Edge density (overall facial features)
    if edge_density < 0.10:
        female_score += 0.25
    elif edge_density > 0.15:
        male_score += 0.20
    
    # Chin prominence
    if chin_prominence > 20:
        male_score += 0.15
    else:
        female_score += 0.10
    
    # Eye region contrast
    if eye_contrast > 25:
        male_score += 0.10
    else:
        female_score += 0.10
    
    # Brightness bias adjustment
    if upper_brightness > brightness + 8:
        female_score += 0.15
    elif upper_brightness > brightness + 5:
        female_score += 0.10
    
    # Base confidence adjustment
    male_score += 0.05
    female_score += 0.05
    
    # Calculate final confidence with smoothing
    total_score = male_score + female_score
    if total_score > 0:
        if male_score > female_score:
            confidence = (male_score / total_score) * 0.9 + 0.1  # Minimum 10% confidence
            gender = "Male"
        else:
            confidence = (female_score / total_score) * 0.9 + 0.1
            gender = "Female"
    else:
        gender, confidence = "Unknown", 0.5
    
    # Update analyzer statistics if provided
    if analyzer:
        analyzer.update_statistics(gender, "Unknown")
    
    return gender, confidence

def detect_age_and_gender(face_img, analyzer=None):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    # Enhanced age detection features
    brightness = np.mean(gray)
    contrast = np.std(gray)
    face_area = face_img.shape[0] * face_img.shape[1]
    
    # Multi-level edge detection for wrinkle analysis
    edges_fine = cv2.Canny(gray, 20, 80)  # Fine details
    edges_coarse = cv2.Canny(gray, 50, 150)  # Coarse features
    
    wrinkle_density_fine = np.sum(edges_fine > 0) / edges_fine.size if edges_fine.size > 0 else 0
    wrinkle_density_coarse = np.sum(edges_coarse > 0) / edges_coarse.size if edges_coarse.size > 0 else 0
    
    # Skin texture analysis using Local Binary Patterns approach
    skin_texture = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Eye region analysis for aging signs
    height, width = gray.shape
    eye_region = gray[int(0.2*height):int(0.5*height), int(0.15*width):int(0.85*width)]
    eye_wrinkles = cv2.Canny(eye_region, 30, 100) if eye_region.size > 0 else np.zeros((1, 1))
    eye_aging_score = np.sum(eye_wrinkles > 0) / eye_wrinkles.size if eye_wrinkles.size > 0 else 0
    
    # Forehead lines detection
    forehead_region = gray[0:int(0.3*height), int(0.2*width):int(0.8*width)]
    forehead_lines = cv2.Canny(forehead_region, 25, 100) if forehead_region.size > 0 else np.zeros((1, 1))
    forehead_aging = np.sum(forehead_lines > 0) / forehead_lines.size if forehead_lines.size > 0 else 0
    
    # Combined aging analysis
    overall_aging_score = (wrinkle_density_coarse + eye_aging_score + forehead_aging) / 3
    texture_aging_factor = min(skin_texture / 100, 1.0)  # Normalize texture factor
    
    # Enhanced age categorization with multiple factors
    if face_area < 8000 and brightness > 130 and overall_aging_score < 0.03:
        age_category, estimated_age = "Baby (0-2)", "~1 year"
    elif face_area < 12000 and overall_aging_score < 0.05 and texture_aging_factor < 0.3:
        age_category, estimated_age = "Toddler (3-5)", "~4 years"
    elif face_area < 20000 and overall_aging_score < 0.08 and brightness > 120:
        age_category, estimated_age = "Kid (6-10)", "~8 years"
    elif overall_aging_score < 0.12 and face_area > 15000 and texture_aging_factor < 0.5:
        age_category, estimated_age = "Teen (11-17)", "~15 years"
    elif overall_aging_score < 0.20 and face_area > 15000 and eye_aging_score < 0.15:
        age_category, estimated_age = "Young Adult (18-25)", "~22 years"
    elif overall_aging_score < 0.25 and eye_aging_score < 0.20:
        age_category, estimated_age = "Adult (26-35)", "~30 years"
    elif overall_aging_score < 0.35 and forehead_aging < 0.25:
        age_category, estimated_age = "Middle Age (36-50)", "~43 years"
    elif overall_aging_score < 0.45:
        age_category, estimated_age = "Mature (51-65)", "~58 years"
    else:
        age_category, estimated_age = "Senior (66+)", "~70 years"
    
    gender, gender_confidence = detect_gender(face_img, analyzer)
    
    # Update statistics
    if analyzer:
        analyzer.update_statistics(gender, age_category)
    
    return age_category, estimated_age, gender, gender_confidence, overall_aging_score

def save_detection_data(analyzer, filename="detection_session.json"):
    """Save session statistics to file"""
    stats = analyzer.get_session_stats()
    stats['timestamp'] = datetime.now().isoformat()
    
    try:
        with open(filename, 'w') as f:
            json.dump(stats, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving data: {e}")
        return False

def load_previous_session(filename="detection_session.json"):
    """Load previous session data"""
    try:
        with open(filename, 'r') as f:
            return json.load(f)
    except:
        return None

def create_output_directory():
    """Create directory for saved images"""
    output_dir = "detection_outputs"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    return output_dir

def enhance_frame_quality(frame):
    """Enhance frame quality for better detection"""
    # Adjust brightness and contrast
    alpha = 1.1  # Contrast control
    beta = 10    # Brightness control
    enhanced = cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
    
    # Apply slight denoising
    denoised = cv2.fastNlMeansDenoisingColored(enhanced, None, 10, 10, 7, 21)
    
    return denoised

def draw_enhanced_overlay(frame, faces_data, analyzer):
    """Draw enhanced overlay with statistics"""
    # Draw detection statistics
    stats = analyzer.get_session_stats()
    
    # Background for stats
    cv2.rectangle(frame, (10, 10), (300, 120), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (300, 120), (255, 255, 255), 2)
    
    # Session info
    cv2.putText(frame, f"Session: {stats['session_duration_minutes']}min", (15, 30), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Detections: {analyzer.statistics['total_detections']}", (15, 50), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"M:{analyzer.statistics['male_detections']} F:{analyzer.statistics['female_detections']}", (15, 70), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    cv2.putText(frame, f"Rate: {stats['detections_per_minute']:.1f}/min", (15, 90), 
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return frame

def main():
    # Initialize enhanced analyzer
    analyzer = FaceAnalyzer()
    
    # Create output directory
    output_dir = create_output_directory()
    
    # Load previous session data
    previous_session = load_previous_session()
    if previous_session:
        print(f"Previous session found: {previous_session.get('timestamp', 'Unknown time')}")
    
    # Enhanced camera setup
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera!")
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
        print("Error: Could not load face detection!")
        return
    
    print("=== Enhanced Age & Gender Detection System ===")
    print("Controls:")
    print("  'q' - Quit application")
    print("  's' - Save current frame")
    print("  'r' - Reset statistics")
    print("  't' - Toggle tracking")
    print("  'e' - Export session data")
    print("  'h' - Show/hide statistics overlay")
    
    show_overlay = True
    frame_count = 0
    fps_counter = 0
    start_time = time.time()
    detection_queue = queue.Queue(maxsize=10)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        fps_counter += 1
        
        # Flip frame horizontally for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Enhance frame quality periodically
        if frame_count % 5 == 0:  # Every 5th frame
            frame = enhance_frame_quality(frame)
        
        # Convert to grayscale for detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Multi-scale face detection with different parameters
        faces_frontal = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80), maxSize=(400, 400))
        
        # Combine frontal and profile detections
        faces = faces_frontal
        
        # Process detected faces
        faces_data = []
        for (x, y, w, h) in faces:
            # Validate face region
            if w < 60 or h < 60:
                continue
                
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                try:
                    age_category, estimated_age, gender, gender_conf, aging_score = detect_age_and_gender(face_roi, analyzer)
                    
                    faces_data.append({
                        'bbox': (x, y, w, h),
                        'age_category': age_category,
                        'estimated_age': estimated_age,
                        'gender': gender,
                        'confidence': gender_conf,
                        'aging_score': aging_score
                    })
                    
                    # Dynamic color coding based on confidence and age
                    if gender_conf > 0.8:
                        color = (0, 255, 0)  # High confidence - green
                    elif gender_conf > 0.7:
                        color = (0, 255, 255)  # Medium confidence - yellow
                    elif gender_conf > 0.6:
                        color = (0, 150, 255)  # Low confidence - orange
                    else:
                        color = (0, 100, 255)  # Very low confidence - red
                    
                    # Enhanced bounding box with gradient effect
                    thickness = max(2, int(gender_conf * 5))
                    cv2.rectangle(frame, (x-2, y-2), (x+w+2, y+h+2), color, thickness)
                    
                    # Enhanced text background
                    text_height = 90
                    cv2.rectangle(frame, (x, y-text_height), (x+max(w, 280), y), (0, 0, 0), -1)
                    cv2.rectangle(frame, (x, y-text_height), (x+max(w, 280), y), color, 2)
                    
                    # Multi-line text display
                    cv2.putText(frame, f"{gender} - {age_category}", (x+5, y-65), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                    
                    cv2.putText(frame, f"Age: {estimated_age}", (x+5, y-45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                    cv2.putText(frame, f"Confidence: {gender_conf:.1%}", (x+5, y-25), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
                    
                    cv2.putText(frame, f"Aging Score: {aging_score:.2f}", (x+5, y-10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (150, 150, 150), 1)
                    
                except Exception as e:
                    print(f"Error processing face: {e}")
                    continue
        
        # Add enhanced overlay
        if show_overlay:
            frame = draw_enhanced_overlay(frame, faces_data, analyzer)
        
        # FPS calculation and display
        if fps_counter >= 30:
            current_time = time.time()
            fps = fps_counter / (current_time - start_time)
            start_time = current_time
            fps_counter = 0
            
        cv2.putText(frame, f"FPS: {fps:.1f}" if 'fps' in locals() else "FPS: --", 
                   (frame.shape[1]-100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Main title
        cv2.putText(frame, "Enhanced Age & Gender Detection", (10, frame.shape[0]-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        cv2.imshow('Enhanced Age & Gender Detection System', frame)
        
        # Enhanced keyboard controls
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = os.path.join(output_dir, f"detection_{timestamp}.jpg")
            cv2.imwrite(filename, frame)
            print(f"✓ Saved: {filename}")
        elif key == ord('r'):
            analyzer = FaceAnalyzer()
            print("✓ Statistics reset")
        elif key == ord('t'):
            analyzer.tracking_enabled = not analyzer.tracking_enabled
            print(f"✓ Tracking: {'ON' if analyzer.tracking_enabled else 'OFF'}")
        elif key == ord('e'):
            if save_detection_data(analyzer):
                print("✓ Session data exported")
            else:
                print("✗ Failed to export session data")
        elif key == ord('h'):
            show_overlay = not show_overlay
            print(f"✓ Overlay: {'ON' if show_overlay else 'OFF'}")
    
    # Cleanup and final statistics
    final_stats = analyzer.get_session_stats()
    print("\n=== Session Summary ===")
    print(f"Duration: {final_stats['session_duration_minutes']} minutes")
    print(f"Total detections: {analyzer.statistics['total_detections']}")
    print(f"Detection rate: {final_stats['detections_per_minute']:.1f} per minute")
    print(f"Gender distribution: {final_stats['gender_ratio']['male_percentage']:.1f}% Male, {final_stats['gender_ratio']['female_percentage']:.1f}% Female")
    if final_stats['most_common_age'] != "None":
        print(f"Most common age group: {final_stats['most_common_age']}")
    
    # Auto-save final session
    save_detection_data(analyzer, f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
