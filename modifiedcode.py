import cv2
import mediapipe as mp

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils

# Start Webcam
cap = cv2.VideoCapture(0)

# Finger tip landmark indices
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

def get_finger_fold_status(lm_list):
    return [lm_list[tip].x < lm_list[tip - 2].x for tip in finger_tips]

while True:
    ret, img = cap.read()
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmark.landmark]
            finger_fold_status = get_finger_fold_status(lm_list)

            # Thumb up or down
            thumb_up = lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y
            thumb_down = lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y

            # STOP
            if all(lm_list[tip].y < lm_list[tip - 2].y for tip in finger_tips) and lm_list[4].y < lm_list[2].y:
                cv2.putText(img, "STOP", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # FORWARD
            if lm_list[3].x > lm_list[4].x and not finger_fold_status[0] and all(finger_fold_status[1:]):
                cv2.putText(img, "FORWARD", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 3)

            # BACKWARD
            if lm_list[3].x > lm_list[4].x and lm_list[3].y < lm_list[4].y and finger_fold_status[0] and not any(finger_fold_status[1:]):
                cv2.putText(img, "BACKWARD", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 100, 0), 3)

            # LEFT
            if lm_list[4].y < lm_list[2].y and finger_fold_status == [False, True, True, True] and lm_list[5].x < lm_list[0].x:
                cv2.putText(img, "LEFT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)

            # RIGHT
            if lm_list[4].y < lm_list[2].y and finger_fold_status == [False, True, True, True] and lm_list[5].x > lm_list[0].x:
                cv2.putText(img, "RIGHT", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 3)

            # LIKE
            if all(finger_fold_status) and thumb_up:
                cv2.putText(img, "LIKE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

            # DISLIKE
            if all(finger_fold_status) and thumb_down:
                cv2.putText(img, "DISLIKE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

            # PEACE
            if finger_fold_status == [False, False, True, True]:
                cv2.putText(img, "PEACE", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)

            # ROCK
            if finger_fold_status == [False, True, True, False]:
                cv2.putText(img, "ROCK", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (200, 100, 255), 3)

            # OKAY
            if abs(lm_list[8].x - lm_list[4].x) < 0.05 and abs(lm_list[8].y - lm_list[4].y) < 0.05:
                cv2.putText(img, "OKAY", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 128, 255), 3)

            # FIST
            if all(finger_fold_status) and not thumb_up and not thumb_down:
                cv2.putText(img, "FIST", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 255), 3)

            # OPEN PALM
            if not any(finger_fold_status) and lm_list[4].y < lm_list[2].y:
                cv2.putText(img, "PALM", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 200, 200), 3)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2),
                                   mp_draw.DrawingSpec(color=(0, 255, 0), thickness=2))

    cv2.imshow("Hand Sign Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # Press ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
