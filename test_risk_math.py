import cv2
import mediapipe as mp
import numpy as np

# Load MediaPipe Tasks dependencies
BaseOptions = mp.tasks.BaseOptions
PoseLandmarker = mp.tasks.vision.PoseLandmarker
PoseLandmarkerOptions = mp.tasks.vision.PoseLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Hardcoded standard MediaPipe pose connections (start_joint, end_joint)
POSE_CONNECTIONS = [
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (9, 10),
    (11, 12),
    (11, 13),
    (13, 15),
    (15, 17),
    (15, 19),
    (15, 21),
    (17, 19),
    (12, 14),
    (14, 16),
    (16, 18),
    (16, 20),
    (16, 22),
    (18, 20),
    (11, 23),
    (12, 24),
    (23, 24),
    (23, 25),
    (24, 26),
    (25, 27),
    (26, 28),
    (27, 29),
    (28, 30),
    (29, 31),
    (30, 32),
    (27, 31),
    (28, 32),
]

# Configure options
options = PoseLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="pose_landmarker_heavy.task"),
    running_mode=VisionRunningMode.VIDEO,
)

cap = cv2.VideoCapture(0)

with PoseLandmarker.create_from_options(options) as landmarker:
    timestamp = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            continue

        # Convert OpenCV BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

        # Detect landmarks
        timestamp += 1
        results = landmarker.detect_for_video(mp_image, timestamp)

        # Draw detected landmarks on frame
        if results.pose_landmarks:
            landmarks = results.pose_landmarks[0]
            h, w, _ = frame.shape

            # 1. Draw connection lines (Skeleton)
            for start_idx, end_idx in POSE_CONNECTIONS:
                pt1 = landmarks[start_idx]
                pt2 = landmarks[end_idx]

                # Safe visibility check handling None values
                v1 = pt1.visibility if pt1.visibility is not None else 1.0
                v2 = pt2.visibility if pt2.visibility is not None else 1.0

                if v1 > 0.5 and v2 > 0.5:
                    x1, y1 = int(pt1.x * w), int(pt1.y * h)
                    x2, y2 = int(pt2.x * w), int(pt2.y * h)
                    cv2.line(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)

            # 2. Draw joint dots
            for landmark in landmarks:
                v = (
                    landmark.visibility
                    if landmark.visibility is not None
                    else 1.0
                )
                if v > 0.5:
                    cx, cy = int(landmark.x * w), int(landmark.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.imshow("MediaPipe Pose Feed", frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

cap.release()
cv2.destroyAllWindows()