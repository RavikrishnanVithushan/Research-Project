import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# Load model
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="models/hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

detector = HandLandmarker.create_from_options(options)

# Webcam
cap = cv2.VideoCapture(0)

while True:

    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=rgb
    )

    results = detector.detect_for_video(mp_image, 0)

    if results.hand_landmarks:
        for hand_landmarks in results.hand_landmarks:
            for landmark in hand_landmarks:
                x = int(landmark.x * frame.shape[1])
                y = int(landmark.y * frame.shape[0])

                cv2.circle(frame, (x, y), 5, (0,255,0), -1)

    cv2.imshow("Hand Detection", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()