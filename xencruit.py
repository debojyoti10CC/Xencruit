import cv2
import mediapipe as mp
import numpy as np
import time
from ultralytics import YOLO


mp_face_mesh = mp.solutions.face_mesh
mp_pose = mp.solutions.pose
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1, min_detection_confidence=0.5)
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

model = YOLO('yolov8n.pt')  


class_names = model.names


print("Available classes in the model:")
for idx, name in class_names.items():
    if 'phone' in name.lower() or 'cell' in name.lower() or 'mobile' in name.lower():
        print(f"Class ID {idx}: {name}")


cheating_objects = {67, 73, 74, 75, 76, 77}  

detection_confidence_threshold = 0.3

cap = cv2.VideoCapture(0)

def draw_grid(frame):
    h, w, _ = frame.shape
    step_size = 40
    for x in range(0, w, step_size):
        cv2.line(frame, (x, 0), (x, h), (50, 50, 50), 1)
    for y in range(0, h, step_size):
        cv2.line(frame, (0, y), (w, y), (50, 50, 50), 1)

def eye_openness(landmarks, indices):
    left = np.linalg.norm(np.array(landmarks[indices[0]]) - np.array(landmarks[indices[1]]))
    right = np.linalg.norm(np.array(landmarks[indices[2]]) - np.array(landmarks[indices[3]]))
    return (left + right) / 2

def estimate_confidence(eye_openness_level, face_tilt, posture_score, blink_rate, cheating_penalty=0):
    confidence = 100 - abs(face_tilt * 5)
    confidence *= eye_openness_level * 10
    confidence *= (posture_score / 100)
    confidence *= (1 - blink_rate / 10)
    confidence -= cheating_penalty  
    return min(max(confidence, 0), 100)

blink_count = 0
blink_time = time.time()
prev_eye_status = "Open"

cheating_detected = False
cheating_start_time = None
total_cheating_time = 0
cheating_penalty = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    pose_results = pose.process(rgb_frame)
    
    detections = model(frame, conf=detection_confidence_threshold)[0]
    phone_detected = False
    cheating_objects_detected = []
    
    for detection in detections.boxes.data.cpu().numpy():
        class_id = int(detection[5])  
        conf = detection[4]
        x1, y1, x2, y2 = int(detection[0]), int(detection[1]), int(detection[2]), int(detection[3])
        
        
        if class_id in cheating_objects:
            object_name = class_names.get(class_id, f"Unknown-{class_id}")
            cheating_objects_detected.append(object_name)
            
         
            if class_id == 67: 
                phone_detected = True
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(frame, f"{object_name} ({conf:.2f})", (x1, y1 - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    
   
    if cheating_objects_detected:
        cheating_detected = True
        if cheating_start_time is None:
            cheating_start_time = time.time()
        cheating_duration = time.time() - cheating_start_time
        total_cheating_time += cheating_duration / 100  
        
        cheating_penalty = min(50, total_cheating_time * 5)
    else:
        cheating_detected = False
        cheating_start_time = None
    
    landmarks = {}
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            for idx, lm in enumerate(face_landmarks.landmark):
                x, y = int(lm.x * w), int(lm.y * h)
                landmarks[idx] = (x, y)
                cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)
            
            left_eye = [33, 160, 159, 158]
            right_eye = [362, 385, 386, 387]
            eye_level = (eye_openness(landmarks, left_eye) + eye_openness(landmarks, right_eye)) / 2
            
            eye_threshold = 5
            eye_status = "Closed" if eye_level < eye_threshold else "Open"
            if eye_status == "Closed" and prev_eye_status == "Open":
                blink_count += 1
            prev_eye_status = eye_status
            blink_rate = blink_count / (time.time() - blink_time)
            
            left_cheek = landmarks.get(234, (0, 0))
            right_cheek = landmarks.get(454, (0, 0))
            face_tilt = (right_cheek[1] - left_cheek[1]) / h * 100
            
            posture_score = 100
            if pose_results.pose_landmarks:
                left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                shoulder_diff = abs(left_shoulder.y - right_shoulder.y)
                posture_score = max(0, 100 - (shoulder_diff * 200))
            
            confidence = estimate_confidence(eye_level, face_tilt, posture_score, blink_rate, cheating_penalty)
            
            bar_x, bar_y = 50, 50
            bar_height = 300
            filled_height = int((confidence / 100) * bar_height)
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + 50, bar_y + bar_height), (100, 100, 100), 2)
            cv2.rectangle(frame, (bar_x, bar_y + bar_height - filled_height), (bar_x + 50, bar_y + bar_height), (0, 255, 0), -1)
            
            cv2.putText(frame, f"Confidence: {int(confidence)}%", (bar_x + 70, bar_y + bar_height - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Blink Rate: {blink_rate:.2f} blinks/sec", (50, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            cv2.putText(frame, f"Posture: {int(posture_score)}%", (50, 450), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
            
    draw_grid(frame)
    
    # Display cheating warnings
    if cheating_detected:
        cv2.putText(frame, "WARNING: CHEATING DETECTED!", (50, 500), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        cv2.putText(frame, f"Objects: {', '.join(cheating_objects_detected)}", (50, 540), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(frame, f"Cheating Penalty: -{int(cheating_penalty)}%", (50, 580), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    cv2.imshow("Interview Confidence Meter", frame)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
