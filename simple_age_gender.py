import cv2
import numpy as np

def guess_gender(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    height, width = gray.shape
    aspect_ratio = width / height
    
    jaw_area = gray[int(height*0.7):, :]
    jaw_edges = cv2.Canny(jaw_area, 50, 150)
    jaw_strength = np.sum(jaw_edges) / jaw_edges.size
    
    overall_contrast = np.std(gray)
    
    male_indicators = 0
    female_indicators = 0
    
    if aspect_ratio > 0.82:
        male_indicators += 1
    else:
        female_indicators += 1
    
    if jaw_strength > 12:
        male_indicators += 1
    else:
        female_indicators += 1
    
    if overall_contrast > 22:
        male_indicators += 1
    else:
        female_indicators += 1
    
    if male_indicators > female_indicators:
        return "Male"
    else:
        return "Female"

def detect_age_simple(face_img):
    gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
    
    brightness = np.mean(gray)
    face_area = face_img.shape[0] * face_img.shape[1]
    
    edges = cv2.Canny(gray, 30, 100)
    wrinkle_density = np.sum(edges > 0) / edges.size
    
    if face_area < 10000 and brightness > 120:
        return "Child", "~8 years"
    elif wrinkle_density < 0.10:
        return "Teen", "~16 years"  
    elif wrinkle_density < 0.20:
        return "Young Adult", "~22 years"
    elif wrinkle_density < 0.30:
        return "Adult", "~35 years"
    else:
        return "Senior", "~55 years"

def main():
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Cannot open camera!")
        return
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    
    print("Press 'q' to quit")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.flip(frame, 1)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5, minSize=(60, 60))
        
        for (x, y, w, h) in faces:
            face = frame[y:y+h, x:x+w]
            
            if face.size > 0:
                age_cat, age_est = detect_age_simple(face)
                gender = guess_gender(face)
                
                if gender == "Male":
                    color = (255, 0, 0)
                else:
                    color = (255, 0, 255)
                
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 3)
                cv2.rectangle(frame, (x, y-60), (x+w, y), (0, 0, 0), -1)
                
                cv2.putText(frame, f"{gender} {age_cat}", (x+5, y-35), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                cv2.putText(frame, age_est, (x+5, y-15), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.putText(frame, "Age & Gender Detection", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow('Age & Gender Detection', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
