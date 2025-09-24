# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("image-feature-extraction", model="nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True)

# Load model directly
from transformers import AutoModel
model = AutoModel.from_pretrained("nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True, torch_dtype="auto")

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from collections import deque
import matplotlib.pyplot as plt
import os
from transformers import pipeline
from numpy.linalg import norm
from PIL import Image

# --- Ensure folders exist ---
NORMAL_DIR = "Normal Videos"
FPS20_DIR = "20 FPS Videos"
LOG_DIR = "log"
os.makedirs(NORMAL_DIR, exist_ok=True)
os.makedirs(FPS20_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# --- Model for embeddings ---
pipe = pipeline("image-feature-extraction",
                model="nomic-ai/nomic-embed-vision-v1.5",
                trust_remote_code=True)

def mse(imageA, imageB):
    return np.mean((imageA.astype("float") - imageB.astype("float")) ** 2)

def cosine_similarity(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

def convert_to_fps(video_path, target_fps=20):
    base_name = os.path.basename(video_path)
    name, ext = os.path.splitext(base_name)
    new_path = os.path.join(FPS20_DIR, f"{name}_{target_fps}fps{ext}")

    if os.path.exists(new_path):
        print(f"✅ Using existing converted file: {new_path}")
        return new_path

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"❌ Could not open video file: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(new_path, fourcc, target_fps, (width, height))

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

def compute_embedding(frame):
    """Compute a single 1D embedding vector for a frame."""
    pil_image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    emb = pipe(pil_image)[0]  # (num_patches, embedding_dim)
    return np.mean(np.array(emb), axis=0)  # (embedding_dim,)

def detect_scene_changes_with_embeddings(video_path, threshold_ssim=0.7, threshold_mse=3000,
                                         base_emb_diff_thresh=0.15, window_size=20, step=1):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")

    results = []
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ret, prev_frame = cap.read()
    if not ret:
        raise ValueError("❌ Could not read first frame")

    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    prev_emb = compute_embedding(prev_frame)
    prev_index = 0
    frame_index = step

    emb_diffs_window = deque(maxlen=window_size)

    while frame_index < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        score_ssim, _ = ssim(prev_gray, gray, full=True)
        score_mse = mse(prev_gray, gray)

        curr_emb = compute_embedding(frame)
        cos_sim = cosine_similarity(prev_emb, curr_emb)
        emb_diff = 1 - cos_sim

        emb_diffs_window.append(emb_diff)

        if len(emb_diffs_window) >= 5:
            mean = np.mean(emb_diffs_window)
            std = np.std(emb_diffs_window)
            adaptive_thresh = max(base_emb_diff_thresh, mean + 2 * std)
        else:
            adaptive_thresh = base_emb_diff_thresh

        if score_ssim < threshold_ssim and score_mse > threshold_mse and emb_diff > adaptive_thresh:
            time_sec = frame_index / fps
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            results.append({
                "frames": (prev_index, frame_index),
                "ssim": score_ssim,
                "mse": score_mse,
                "emb_diff": emb_diff,
                "time": timestamp
            })

        prev_gray = gray
        prev_emb = curr_emb
        prev_index = frame_index
        frame_index += step

    cap.release()
    return results

def log_results(log_file, video_name, results):
    with open(log_file, "a", encoding="utf-8") as f:
        f.write(f"\nVideo: {video_name}\n")
        if not results:
            f.write("  No significant cuts detected.\n")
        else:
            for r in results:
                f1, f2 = r["frames"]
                f.write(f"  Cut {f1}-{f2} | SSIM={r['ssim']:.3f}, "
                        f"MSE={r['mse']:.1f}, EmbDiff={r['emb_diff']:.3f} | "
                        f"Time={r['time']}\n")
        f.write("-" * 50 + "\n")

def display_all_scene_changes(video_path, scene_changes, max_scenes=5):
    """
    Display before, cut, and after frames for detected scene changes.
    Each frame shows its own metrics and timestamp.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("❌ Could not open video file")

    fps = cap.get(cv2.CAP_PROP_FPS)
    n_scenes = min(len(scene_changes), max_scenes)
    fig, axes = plt.subplots(n_scenes, 3, figsize=(15, 5 * n_scenes))

    if n_scenes == 1:
        axes = [axes]  # make iterable if only one row

    for idx, change in enumerate(scene_changes[:n_scenes]):
        f1, f2 = change["frames"]

        for j, frame_idx in enumerate([f1, f2, f2 + 1]):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Compute metrics against previous frame for each displayed image
            if j == 0:  # Before
                ref_idx = f1 - 1 if f1 > 0 else f1
            else:       # Cut and After
                ref_idx = frame_idx - 1

            cap.set(cv2.CAP_PROP_POS_FRAMES, ref_idx)
            _, ref_frame = cap.read()
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            score_ssim, _ = ssim(ref_gray, curr_gray, full=True)
            score_mse = mse(ref_gray, curr_gray)
            emb_diff = 1 - cosine_similarity(
                compute_embedding(ref_frame),
                compute_embedding(frame)
            )

            time_sec = frame_idx / fps
            minutes = int(time_sec // 60)
            seconds = int(time_sec % 60)
            timestamp = f"{minutes:02d}:{seconds:02d}"

            axes[idx][j].imshow(frame_rgb)
            axes[idx][j].axis("off")
            axes[idx][j].set_title(
                f"{'Before' if j==0 else 'Cut' if j==1 else 'After'} Frame\n"
                f"SSIM={score_ssim:.3f}, MSE={score_mse:.1f}\n"
                f"EmbDiff={emb_diff:.3f}, Time={timestamp}",
                fontsize=9
            )

    plt.tight_layout()
    plt.show()
    cap.release()

# --- Usage ---
if __name__ == "__main__":
    video_path = os.path.join(NORMAL_DIR, "This Is What Peak Leftism Looks Like 🙄.mp4")
    log_file = os.path.join(LOG_DIR, "scene_changes.log")

    converted_video = convert_to_fps(video_path, target_fps=20)
    scene_changes = detect_scene_changes_with_embeddings(converted_video)

    print("Detected significant hard cuts (embedding-aware):")
    for r in scene_changes:
        f1, f2 = r["frames"]
        print(f"Cut {f1}-{f2} | SSIM={r['ssim']:.3f}, "
                f"MSE={r['mse']:.1f}, EmbDiff={r['emb_diff']:.3f} | "
                f"Time={r['time']}")

    log_results(log_file, os.path.basename(video_path), scene_changes)
    print(f"📄 Results logged to {log_file}")

    # Display
    display_all_scene_changes(converted_video, scene_changes, max_scenes=5)
