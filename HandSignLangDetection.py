import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_draw = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Finger tip landmark indices
finger_tips = [8, 12, 16, 20]
thumb_tip = 4

# Helper to load and resize images safely
def load_gesture_image(path, size=(200, 180)):
    img = cv2.imread(path)
    if img is None:
        print(f"❌ Error: Could not load image from {path}. Please check the path or file.")
        return None
    return cv2.resize(img, size)

# Load gesture images safely
like_img = load_gesture_image("images/like.png")
dislike_img = load_gesture_image("images/dislike.png")
peace_img = load_gesture_image("images/peace.png")
rock_img = load_gesture_image("images/rock.png")

def is_finger_up(lm_list, tip_id):
    return lm_list[tip_id].y < lm_list[tip_id - 2].y

while True:
    ret, img = cap.read()
    if not ret:
        break

    img = cv2.flip(img, 1)
    h, w, c = img.shape
    results = hands.process(img)

    if results.multi_hand_landmarks:
        for hand_landmark in results.multi_hand_landmarks:
            lm_list = [lm for lm in hand_landmark.landmark]

            finger_status = [is_finger_up(lm_list, tip) for tip in finger_tips]
            thumb_up = lm_list[thumb_tip].y < lm_list[thumb_tip - 1].y < lm_list[thumb_tip - 2].y
            thumb_down = lm_list[thumb_tip].y > lm_list[thumb_tip - 1].y > lm_list[thumb_tip - 2].y

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmark, mp_hands.HAND_CONNECTIONS,
                                   mp_draw.DrawingSpec((0, 0, 255), 6, 3),
                                   mp_draw.DrawingSpec((0, 255, 0), 4, 2))

            # LIKE
            if all(finger_status) and thumb_up and like_img is not None:
                cv2.putText(img, "LIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
                img[35:215, 30:230] = like_img

            # DISLIKE
            elif all(finger_status) and thumb_down and dislike_img is not None:
                cv2.putText(img, "DISLIKE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                img[35:215, 30:230] = dislike_img

            # PEACE (index and middle up)
            elif finger_status[0] and finger_status[1] and not finger_status[2] and not finger_status[3] and peace_img is not None:
                cv2.putText(img, "PEACE", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 3)
                img[35:215, 30:230] = peace_img

            # ROCK (index and pinky up)
            elif finger_status[0] and not finger_status[1] and not finger_status[2] and finger_status[3] and rock_img is not None:
                cv2.putText(img, "ROCK", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 255), 3)
                img[35:215, 30:230] = rock_img

    cv2.imshow("Hand Sign Detection", img)
    if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
        break

cap.release()
cv2.destroyAllWindows()
