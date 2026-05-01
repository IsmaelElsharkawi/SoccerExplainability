import argparse
import json
import os
import cv2

JSON_PATH = "train_data/json/one_video.json"
DEFAULT_VIDEOS_BASE = (
    "/mnt/c/Users/z004kjmt/Downloads"
    "/SoccerLensVisualizations-no-bboxes-20260421T231816Z-3-001_v2"
    "/SoccerLensVisualizations-no-bboxes"
)
OUTPUT_DIR = "extracted_frames"

MODELS = ["MatchVision", "SigLip", "SoccerMaster"]
METHODS = ["Chefer", "GradCam"]

METHOD_TAG = {
    "Chefer": "chefer",
    "GradCam": "gradcam",
}

MODEL_TAG = {
    "MatchVision": "MatchVision",
    "SigLip": "SigLip",
    "SoccerMaster": "SoccerMaster",
}


def extract_frame(video_path, time_sec):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  ERROR: cannot open {video_path}")
        return None
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        print(f"  ERROR: invalid fps for {video_path}")
        cap.release()
        return None
    frame_idx = int(time_sec * fps)
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    cap.release()
    if not ret:
        print(f"  ERROR: cannot read frame {frame_idx} from {video_path}")
        return None
    h, w = frame.shape[:2]
    frame = frame[0:w, 0:w]
    return frame


def main():
    parser = argparse.ArgumentParser(
        description="Extract single frames from visualization videos for paper figures."
    )
    parser.add_argument(
        "--videos_base",
        default=DEFAULT_VIDEOS_BASE,
        help="Root directory containing <Model>/<Method>/<match>/<clip>.mp4",
    )
    args = parser.parse_args()

    with open(JSON_PATH, "r") as f:
        entries = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in entries:
        video_rel = entry["video"]
        caption = entry["caption"]
        vis_time = entry["visualization_time"]

        # The video files are stored under <match>/<clip>.mp4
        # JSON path: league/match/clip.mp4 -> match key is match/clip.mp4
        parts = video_rel.replace("\\", "/").split("/")
        match_clip = "/".join(parts[1:])  # e.g. "2016-12-04 - .../2_20_48.mp4"

        caption_tag = caption.replace(" ", "_")

        for model in MODELS:
            for method in METHODS:
                video_path = os.path.join(args.videos_base, model, method, match_clip)
                if not os.path.exists(video_path):
                    print(f"  MISSING: {video_path}")
                    continue

                frame = extract_frame(video_path, vis_time)
                if frame is None:
                    continue

                out_name = f"{caption_tag}_{METHOD_TAG[method]}_{MODEL_TAG[model]}.png"
                out_path = os.path.join(OUTPUT_DIR, out_name)
                cv2.imwrite(out_path, frame)
                print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
