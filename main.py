import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import os

# --- Ensure folders exist ---
NORMAL_DIR = "Normal Videos"
FPS20_DIR = "20 FPS Videos"
LOG_DIR = "log"
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(FPS20_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

def mse(imageA, imageB):
    # Mean Squared Error
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def convert_to_fps(video_path, target_fps=20):
    """Convert video to target FPS and save in 20 FPS Videos folder (if not exists)."""
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    new_path = os.path.join(FPS20_DIR, f"{name}_{target_fps}fps{ext}")

    # If already exists, reuse
    if os.path.exists(new_path):
        print(f"✅ Using existing converted file: {new_path}")
        return new_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_path, fourcc, target_fps, (width, height))

    # Get original FPS
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(round(original_fps / target_fps)))

    frame_index = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            out.write(frame)
        frame_index += 1

    cap.release()
    out.release()
    print(f"💾 Saved converted video: {new_path}")
    return new_path

def detect_scene_changes(video_path, threshold_ssim=0.7, threshold_mse=3000, step=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")

    results = {}  # key = second, value = (frames, ssim, mse, timestamp)

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
            sec_bucket = int(time_sec)  # bucket per second
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            # Keep the most significant change for this second
            if sec_bucket not in results:
                results[sec_bucket] = ((prev_index, frame_index), score_ssim, score_mse, timestamp)
            else:
                _, existing_ssim, existing_mse, _ = results[sec_bucket]
                if score_ssim < existing_ssim or score_mse > existing_mse:
                    results[sec_bucket] = ((prev_index, frame_index), score_ssim, score_mse, timestamp)

        prev_gray = gray
        prev_index = frame_index
        frame_index += step

    cap.release()
    return list(results.values())

def log_results(log_file, video_name, results):
    """Append results to log file (create if not exists)."""
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nVideo: {video_name}\n")
        if not results:
            f.write("  No significant cuts detected.\n")
        else:
            for (f1, f2), score_ssim, score_mse, timestamp in results:
                f.write(f"  Cut {f1}-{f2} | SSIM={score_ssim:.3f}, MSE={score_mse:.1f} | Time={timestamp}\n")
        f.write("-" * 50 + "\n")

# --- Usage ---
if __name__ == "__main__":
    video_path = os.path.join(NORMAL_DIR, "This Is What Peak Leftism Looks Like 🙄.mp4")
    log_file = os.path.join(LOG_DIR, "scene_changes.log")

    # 1. Convert video to 20 FPS (only if not already converted)
    converted_video = convert_to_fps(video_path, target_fps=20)

    # 2. Run scene detection on converted video
    scene_changes = detect_scene_changes(converted_video, threshold_ssim=0.7, threshold_mse=3000, step=1)

    print("Detected significant hard cuts (one per second):")
    for (f1, f2), score_ssim, score_mse, timestamp in scene_changes:
        print(f"Cut between frame {f1}-{f2} | SSIM={score_ssim:.3f}, MSE={score_mse:.1f} | Time={timestamp}")

    # 3. Log results
    log_results(log_file, os.path.basename(video_path), scene_changes)
    print(f"📄 Results logged to {log_file}")
