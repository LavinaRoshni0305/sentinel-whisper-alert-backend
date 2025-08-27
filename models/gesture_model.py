import cv2
import mediapipe as mp

def run_gesture():
    print("[GESTURE MODEL] Running...")
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)
    cap = cv2.VideoCapture(0)

    def count_fingers(landmarks):
        tips = [4, 8, 12, 16, 20]
        fingers = []
        if landmarks[tips[0]].x < landmarks[tips[0] - 1].x:
            fingers.append(1)
        else:
            fingers.append(0)
        for i in range(1, 5):
            fingers.append(1 if landmarks[tips[i]].y < landmarks[tips[i] - 2].y else 0)
        return fingers.count(1)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                fingers = count_fingers(hand_landmarks.landmark)
                if fingers == 3:  # Example gesture
                    print("🚨 Emergency Triggered via Hand Gesture!")
                    cap.release()
                    cv2.destroyAllWindows()
                    return "gesture_detected"

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return "no_gesture"
