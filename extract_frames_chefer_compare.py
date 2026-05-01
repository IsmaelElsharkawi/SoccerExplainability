import json
import os
import cv2

JSON_PATH = "train_data/json/one_video.json"
OUTPUT_DIR = "extracted_frames_chefer_compare"

# Map (model, method) -> directory produced by the corresponding .sh script.
# Paths are relative to the repo root and match the .sh files' OUTPUT_DIR
# defaults.  Each directory contains <match_name>/<clip>.mp4.
SOURCES = {
    ("MatchVision",  "Chefer-Spatial"):  "output_chefer_matchvision_spatial_only",
    ("MatchVision",  "Chefer-Temporal"): "output_chefer_matchvision_temporal",
    ("SoccerMaster", "Chefer-Spatial"):  "output_chefer_soccermaster_spatial_only",
    ("SoccerMaster", "Chefer-Temporal"): "output_chefer_soccermaster_temporal",
}

METHOD_TAG = {
    "Chefer-Spatial":  "chefer-spatial",
    "Chefer-Temporal": "chefer-temporal",
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
    with open(JSON_PATH, "r") as f:
        entries = json.load(f)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for entry in entries:
        video_rel = entry["video"]
        caption = entry["caption"]
        vis_time = entry["visualization_time"]

        # JSON path: league/match/clip.mp4 -> drop the league prefix.
        parts = video_rel.replace("\\", "/").split("/")
        match_clip = "/".join(parts[1:])

        caption_tag = caption.replace(" ", "_")

        for (model, method), source_dir in SOURCES.items():
            video_path = os.path.join(source_dir, match_clip)
            if not os.path.exists(video_path):
                print(f"  MISSING: {video_path}")
                continue

            frame = extract_frame(video_path, vis_time)
            if frame is None:
                continue

            out_name = f"{caption_tag}_{METHOD_TAG[method]}_{model}.png"
            out_path = os.path.join(OUTPUT_DIR, out_name)
            cv2.imwrite(out_path, frame)
            print(f"  Saved: {out_path}")


if __name__ == "__main__":
    main()
