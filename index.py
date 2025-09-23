import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

def mse(imageA, imageB):
    # Mean Squared Error
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def detect_scene_changes(video_path, threshold_ssim=0.7, threshold_mse=1500, step=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")

    results = []  # store ((prev_frame, curr_frame), SSIM, MSE, timestamp)

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read first frame
    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("❌ Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_index = 0
    frame_index = step

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Compute SSIM + MSE
        score_ssim, _ = ssim(prev_gray, gray, full=True)
        score_mse = mse(prev_gray, gray)

        # Hybrid check
        if score_ssim < threshold_ssim and score_mse > threshold_mse:
            time_sec = frame_index / fps
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            results.append(((prev_index, frame_index), score_ssim, score_mse, timestamp))

        prev_gray = gray
        prev_index = frame_index
        frame_index += step

    cap.release()
    return results


# --- Usage ---
video_path = "video.mp4"
scene_changes = detect_scene_changes(video_path, threshold_ssim=0.7, threshold_mse=1500, step=1)

print("Detected hard cuts (SSIM + MSE):")
for (f1, f2), score_ssim, score_mse, timestamp in scene_changes:
    print(f"Cut between frame {f1}-{f2} | SSIM={score_ssim:.3f}, MSE={score_mse:.1f} | Time={timestamp}")
