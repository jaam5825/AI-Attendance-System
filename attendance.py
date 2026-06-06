import cv2
from deepface import DeepFace
import pandas as pd
from datetime import datetime
import os
import threading

# ── Student registry ──────────────────────────────────────────────────────────
# Tuple: (Reg No, Full Name, Class)
student_registry = {
    "Kareem": ("2023-uam-2320", "Abdul Kareem", "BS IT (6th)"),
    "Qamar": ("2023-uam-2357", "Qamar Iqbal", "BS IT (6th)"),
    "Areeb": ("2023-uam-2187", "Areeb-ul-Hassan", "BS IT (6th)"),
    "Imran": ("2023-uam-2340", "Muhammad Imran", "BS IT (6th)"),
    # "FolderName": ("REG-XXXX-XXX", "Full Name", "Class"),
}

# ── CSV setup ─────────────────────────────────────────────────────────────────
file = "attendance.csv"
COLUMNS = ["Reg.No", "Name", "Class", "Status", "Date", "Time"]

if not os.path.exists(file):
    pd.DataFrame(columns=COLUMNS).to_csv(file, index=False)
else:
    try:
        existing = pd.read_csv(file)
        if list(existing.columns) != COLUMNS:
            backup = file.replace(".csv", "_backup.csv")
            existing.to_csv(backup, index=False)
            print(f"⚠️ Old CSV backed up to {backup}. Creating new format.")
            pd.DataFrame(columns=COLUMNS).to_csv(file, index=False)
    except Exception:
        pd.DataFrame(columns=COLUMNS).to_csv(file, index=False)


# FIX 4: Corrected column name from "Registration_No" → "Reg.No"
# FIX 5: Now accepts and saves class_name too
def mark_attendance(reg_no, student_name, class_name):
    try:
        df = pd.read_csv(file)
    except Exception:
        df = pd.DataFrame(columns=COLUMNS)

    if list(df.columns) != COLUMNS:
        df = pd.DataFrame(columns=COLUMNS)

    today = datetime.now().strftime("%Y-%m-%d")

    # FIX 4: was checking "Registration_No" which doesn't exist — now "Reg.No"
    already_marked = (
        (df["Reg.No"] == reg_no) &
        (df["Date"] == today)
    ).any()

    if not already_marked:
        time_now = datetime.now().strftime("%H:%M:%S")
        new_row = pd.DataFrame(
            [[reg_no, student_name, class_name, "Present", today, time_now]],
            columns=COLUMNS
        )
        new_row.to_csv(file, mode='a', header=False, index=False)
        print(f"✅ Attendance marked — {student_name} ({reg_no}) at {today} {time_now}")
        return True

    print(f"ℹ️ {student_name} ({reg_no}) already marked today.")
    return False


# FIX 1: verify_face_identity removed as a separate step.
# DeepFace.find() already does the comparison against all images.
# Running verify() on top is redundant and doubles the processing time.
def is_known_face(frame, threshold=0.35):
    try:
        result = DeepFace.find(
            img_path=frame,
            db_path="dataset",
            enforce_detection=False,
            model_name="Facenet",
            silent=True,
            detector_backend='opencv'
        )

        if not (result is not None and len(result) > 0 and len(result[0]) > 0):
            print("❌ No matches found in database")
            return False, None, None, None, None, None

        df_matches = result[0].sort_values(by=["distance"])
        top_matches = df_matches.head(3)

        best_match = top_matches.iloc[0]
        best_distance = float(best_match["distance"])
        best_identity = best_match["identity"]
        best_folder_name = os.path.basename(os.path.dirname(best_identity))

        print(f"Best match: {best_folder_name} | distance: {best_distance:.4f}")

        if best_distance >= threshold:
            print(f"❌ Distance {best_distance:.4f} exceeds threshold {threshold}")
            return False, None, None, None, None, None

        # Gap check only triggers for DIFFERENT people
        if len(top_matches) > 1:
            second_identity = top_matches.iloc[1]["identity"]
            second_folder_name = os.path.basename(os.path.dirname(second_identity))
            second_distance = float(top_matches.iloc[1]["distance"])
            gap = second_distance - best_distance
            print(f"Gap to second best: {gap:.4f}")
            if second_folder_name != best_folder_name and gap < 0.08:
                print("❌ Confusion detected — different person too close")
                return False, None, None, None, None, None

        if best_folder_name not in student_registry:
            print(f"❌ '{best_folder_name}' not in student registry — treated as Unknown")
            return False, None, None, None, None, None

        # FIX 5: Unpack all 3 values from registry
        reg_no, student_name, class_name = student_registry[best_folder_name]
        print(f"✅ Confirmed: {student_name} ({reg_no}) | distance: {best_distance:.4f}")
        return True, best_folder_name, reg_no, student_name, class_name, best_distance

    except Exception as e:
        print(f"Recognition error: {e}")
        return False, None, None, None, None, None


def ensure_good_lighting(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    if gray.mean() < 50:
        cv2.putText(frame, "Poor Lighting!", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        return False
    return True


# ── Main loop ─────────────────────────────────────────────────────────────────
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# FIX 2 & 3: Threading — recognition runs in background so camera never freezes
recognition_running = False   # Flag: is a recognition thread active?
recognition_result = None     # Shared result from background thread
result_lock = threading.Lock()

frame_count = 0
recognition_cooldown = 0
last_recognized_name = None
consecutive_matches = 0

# Display state (persists across frames so text stays visible)
display_name = ""
display_reg = ""
display_class = ""
display_confidence = ""
display_status = ""   # "marked", "already", "unknown", "confirming"


def run_recognition(frame_copy):
    """Runs in a background thread — writes result to shared variable."""
    global recognition_result, recognition_running
    result = is_known_face(frame_copy)
    with result_lock:
        recognition_result = result
    recognition_running = False


print("Starting attendance system...")
print("Press Q to quit")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Camera error")
        break

    frame_count += 1
    recognition_cooldown = max(0, recognition_cooldown - 1)
    ensure_good_lighting(frame)

    # Start a recognition thread every 10 frames if none is running
    if frame_count % 10 == 0 and not recognition_running and recognition_cooldown == 0:
        recognition_running = True
        frame_copy = frame.copy()
        t = threading.Thread(target=run_recognition, args=(frame_copy,), daemon=True)
        t.start()

    # Check if a recognition result is ready
    with result_lock:
        if recognition_result is not None:
            is_known, folder_name, reg_no, student_name, class_name, distance = recognition_result
            recognition_result = None

            if is_known and student_name:
                if folder_name == last_recognized_name:
                    consecutive_matches += 1
                else:
                    consecutive_matches = 1
                    last_recognized_name = folder_name

                if consecutive_matches >= 2:
                    status = mark_attendance(reg_no, student_name, class_name)
                    display_name = student_name
                    display_reg = reg_no
                    display_class = class_name
                    display_confidence = f"{((1 - distance) * 100):.1f}%"
                    display_status = "marked" if status else "already"
                    if status:
                        recognition_cooldown = 999  # Stop recognizing after success
                else:
                    display_name = student_name
                    display_status = "confirming"
            else:
                consecutive_matches = 0
                last_recognized_name = None
                display_status = "unknown"
                display_name = ""

    # ── Draw overlays based on current display state ──────────────────────────
    if display_status == "marked":
        cv2.putText(frame, f"Name: {display_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f"Reg No: {display_reg}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, f"Class: {display_class}", (50, 128),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Confidence: {display_confidence}", (50, 163),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Attendance Marked Successfully!", (50, 200),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    elif display_status == "already":
        cv2.putText(frame, f"Name: {display_name}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
        cv2.putText(frame, f"Reg No: {display_reg}", (50, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(frame, "Already Marked Today", (50, 130),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)

    elif display_status == "confirming":
        cv2.putText(frame, f"Recognizing: {display_name} (Confirming...)", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    elif display_status == "unknown":
        cv2.putText(frame, "UNKNOWN PERSON", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(frame, "Access Denied!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

    if recognition_running:
        cv2.putText(frame, "Scanning...", (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (180, 180, 180), 1)

    cv2.imshow("Attendance System", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    # Auto-close after successful marking (cooldown maxed out)
    if recognition_cooldown == 999:
        cv2.waitKey(2000)
        break

cap.release()
cv2.destroyAllWindows()