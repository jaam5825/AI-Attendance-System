import cv2
import os
import time

person_name = "Imran"
save_path = f"dataset/{person_name}"

if not os.path.exists(save_path):
    os.makedirs(save_path)

cap = cv2.VideoCapture(0)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

angles = [
    "Look Straight",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Look Down"
    "Look Straight",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Look Down"
]

count = 0
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

zone_x1 = frame_w // 4
zone_y1 = frame_h // 6
zone_x2 = 3 * frame_w // 4
zone_y2 = 5 * frame_h // 6

COUNTDOWN_SECONDS = 5
face_in_position_since = None  # Timestamp when face first entered correct position

print("Green border = face in correct position | Red border = adjust your position")
print("Hold position for 5 seconds to auto-capture")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(80, 80)
    )

    face_in_position = False

    if len(faces) > 0:
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        x, y, w, h = faces[0]

        face_cx = x + w // 2
        face_cy = y + h // 2

        if zone_x1 < face_cx < zone_x2 and zone_y1 < face_cy < zone_y2:
            face_in_position = True
            border_color = (0, 255, 0)

            # Start countdown timer if not already started
            if face_in_position_since is None:
                face_in_position_since = time.time()

            # Calculate remaining seconds
            elapsed = time.time() - face_in_position_since
            remaining = COUNTDOWN_SECONDS - elapsed

            if remaining > 0:
                # Show countdown on screen
                countdown_text = f"Capturing in: {int(remaining) + 1}s"
                cv2.putText(frame, countdown_text, (x, y + h + 25),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Draw a progress bar below the face
                bar_width = w
                filled = int(bar_width * (elapsed / COUNTDOWN_SECONDS))
                bar_y = y + h + 45
                cv2.rectangle(frame, (x, bar_y), (x + bar_width, bar_y + 10), (50, 50, 50), -1)
                cv2.rectangle(frame, (x, bar_y), (x + filled, bar_y + 10), (0, 255, 0), -1)
            else:
                # Countdown complete — auto capture
                img_path = f"{save_path}/{count}.jpg"
                cv2.imwrite(img_path, frame)
                print(f"✅ Auto-captured: {angles[count]} -> {img_path}")
                count += 1
                face_in_position_since = None  # Reset for next angle

                # Brief flash effect
                flash = frame.copy()
                cv2.rectangle(flash, (0, 0), (frame_w, frame_h), (255, 255, 255), -1)
                cv2.addWeighted(flash, 0.4, frame, 0.6, 0, frame)
                cv2.imshow("Angle Capture", frame)
                cv2.waitKey(800)

        else:
            border_color = (0, 0, 255)
            face_in_position_since = None  # Reset if face leaves zone
            cv2.putText(frame, "Move to center", (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x + w, y + h), border_color, 3)

    else:
        face_in_position_since = None  # Reset if no face detected
        cv2.putText(frame, "No face detected", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    # Guide zone
    cv2.rectangle(frame, (zone_x1, zone_y1), (zone_x2, zone_y2), (200, 200, 200), 1)

    # Angle instruction
    if count < len(angles):
        cv2.putText(frame, f"Step {count + 1}/5: {angles[count]}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Angle Capture", frame)

    if count == len(angles):
        print("✔ All 5 angles captured successfully!")
        cv2.waitKey(1500)
        break

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()