import cv2
import numpy as np
import time

def detect_gender(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    face_height, face_width = gray.shape
    aspect_ratio = face_width / face_height
    
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / edges.size
    
    height_third = face_height // 3
    
    upper_region = gray[0:height_third, :]
    upper_brightness = np.mean(upper_region)
    
    middle_region = gray[height_third:2*height_third, :]
    middle_contrast = np.std(middle_region)
    
    lower_region = gray[2*height_third:, :]
    lower_edges = cv2.Canny(lower_region, 50, 150)
    jaw_definition = np.sum(lower_edges > 0) / lower_edges.size
    
    male_score = 0
    female_score = 0
    
    if jaw_definition > 0.15:
        male_score += 0.4
    else:
        female_score += 0.2
    
    if aspect_ratio > 0.85:
        male_score += 0.3
    else:
        female_score += 0.3
    
    if middle_contrast > 25:
        male_score += 0.3
    else:
        female_score += 0.2
    
    if edge_density < 0.12:
        female_score += 0.3
    else:
        male_score += 0.2
    
    if upper_brightness > brightness + 5:
        female_score += 0.2
    
    male_score += 0.1
    female_score += 0.1
    
    if male_score > female_score:
        confidence = male_score / (male_score + female_score)
        return "Male", confidence
    else:
        confidence = female_score / (male_score + female_score)
        return "Female", confidence

def detect_age_and_gender(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(gray)
    contrast = np.std(gray)
    face_area = face_img.shape[0] * face_img.shape[1]
    
    edges = cv2.Canny(gray, 30, 100)
    wrinkle_density = np.sum(edges > 0) / edges.size if edges.size > 0 else 0
    
    if face_area < 8000 and brightness > 130 and wrinkle_density < 0.03:
        age_category, estimated_age = "Baby (0-2)", "~1 year"
    elif face_area < 12000 and wrinkle_density < 0.05:
        age_category, estimated_age = "Toddler (3-5)", "~4 years"
    elif face_area < 20000 and wrinkle_density < 0.08:
        age_category, estimated_age = "Kid (6-10)", "~8 years"
    elif wrinkle_density < 0.12 and face_area > 15000:
        age_category, estimated_age = "Teen (11-17)", "~15 years"
    elif wrinkle_density < 0.25 and face_area > 15000:
        age_category, estimated_age = "Young Adult (18-25)", "~22 years"
    elif wrinkle_density < 0.30:
        age_category, estimated_age = "Adult (26-40)", "~32 years"
    elif wrinkle_density < 0.40:
        age_category, estimated_age = "Middle Age (41-55)", "~48 years"
    else:
        age_category, estimated_age = "Senior (56+)", "~65 years"
    
    gender, gender_confidence = detect_gender(face_img)
    
    return age_category, estimated_age, gender, gender_confidence

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Cannot access camera!")
        return
    
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    if face_cascade.empty():
        print("Error: Could not load face detection!")
        return
    
    print("Press 'q' to quit, 's' to save")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(80, 80))
        
        for (x, y, w, h) in faces:
            face_roi = frame[y:y+h, x:x+w]
            
            if face_roi.size > 0:
                age_category, estimated_age, gender, gender_conf = detect_age_and_gender(face_roi)
                
                if gender_conf > 0.7:
                    color = (0, 255, 0)
                elif gender_conf > 0.6:
                    color = (0, 255, 255)
                else:
                    color = (0, 100, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                
                cv2.rectangle(frame, (x, y-75), (x+max(w, 220), y), (0, 0, 0), -1)
                
                cv2.putText(frame, f"{gender} {age_category}", (x+5, y-55), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(frame, estimated_age, (x+5, y-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                
                cv2.putText(frame, f"Confidence: {gender_conf:.0%}", (x+5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1)
        
        cv2.putText(frame, "Age & Gender Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Age & Gender Detection', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            filename = f"detection_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved: {filename}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
