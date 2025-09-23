import cv2
import os
import sys

# 🔧 Add parent folder to sys.path so we can import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import detect_scene_changes  # import from main.py

FPS10_DIR = "10 FPS Videos"

def test_scene_detection(video_path):
    """Check that detected cuts <= video duration (in seconds)."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video file: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_seconds = int(total_frames / fps)
    cap.release()

    results = detect_scene_changes(video_path, threshold_ssim=0.6, threshold_mse=4500, step=1)

    if len(results) <= duration_seconds:
        print(f"✅ PASS: {os.path.basename(video_path)} → {len(results)} cuts (video length {duration_seconds} sec)")
    else:
        print(f"❌ FAIL: {os.path.basename(video_path)} → {len(results)} cuts, "
                f"but video length is only {duration_seconds} sec")

if __name__ == "__main__":
    if not os.path.exists(FPS10_DIR):
        print(f"❌ Folder '{FPS10_DIR}' not found.")
    else:
        videos = [f for f in os.listdir(FPS10_DIR) if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))]
        if not videos:
            print(f"⚠️ No converted videos found in '{FPS10_DIR}'")
        else:
            for video in videos:
                video_path = os.path.join(FPS10_DIR, video)
                test_scene_detection(video_path)
