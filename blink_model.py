import cv2
import mediapipe as mp
import time
import math

mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def euclidean_distance(p1, p2):
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])

def calculate_ear(landmarks, eye_indices, w, h):
    p1 = (int(landmarks[eye_indices[0]].x * w), int(landmarks[eye_indices[0]].y * h))
    p2 = (int(landmarks[eye_indices[1]].x * w), int(landmarks[eye_indices[1]].y * h))
    p3 = (int(landmarks[eye_indices[2]].x * w), int(landmarks[eye_indices[2]].y * h))
    p4 = (int(landmarks[eye_indices[3]].x * w), int(landmarks[eye_indices[3]].y * h))
    p5 = (int(landmarks[eye_indices[4]].x * w), int(landmarks[eye_indices[4]].y * h))
    p6 = (int(landmarks[eye_indices[5]].x * w), int(landmarks[eye_indices[5]].y * h))
    return (euclidean_distance(p2, p6) + euclidean_distance(p3, p5)) / (2.0 * euclidean_distance(p1, p4))

def run_blink():
    EAR_THRESHOLD = 0.21
    CONSEC_FRAMES = 3
    cap = cv2.VideoCapture(0)
    blink_counter, closed_frames = 0, 0
    start_time = time.time()

    print("[BLINK MODEL] Running...")

    with mp_face_mesh.FaceMesh(
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3
    ) as face_mesh:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            h, w = frame.shape[:2]

            if results.multi_face_landmarks:
                for landmarks in results.multi_face_landmarks:
                    mp_drawing.draw_landmarks(frame, landmarks, mp_face_mesh.FACEMESH_CONTOURS)
                    left_ear = calculate_ear(landmarks.landmark, LEFT_EYE, w, h)
                    right_ear = calculate_ear(landmarks.landmark, RIGHT_EYE, w, h)
                    avg_ear = (left_ear + right_ear) / 2

                    if avg_ear < EAR_THRESHOLD:
                        closed_frames += 1
                    else:
                        if closed_frames >= CONSEC_FRAMES:
                            blink_counter += 1
                            print(f"[BLINK MODEL] Blink #{blink_counter}")
                        closed_frames = 0

                    if time.time() - start_time < 5 and blink_counter >= 3:
                        print("🚨 Emergency Triggered via Blink Pattern!")
                        cap.release()
                        cv2.destroyAllWindows()
                        return "blink_detected"

            cv2.imshow("Blink Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()
    return "no_blink"
