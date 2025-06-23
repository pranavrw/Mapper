import cv2
import torch
import easyocr
import numpy as np
import pandas as pd
from ultralytics import YOLO
from torchvision import models, transforms
from deep_sort_realtime.deepsort_tracker import DeepSort
from scipy.spatial.distance import cosine
from collections import defaultdict
import os

# ---------------------- Setup and Model Loading ----------------------


device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")


model = YOLO("best.pt")
model.to(device)

reid_model = models.resnet50(pretrained=True)
reid_model.fc = torch.nn.Identity()
reid_model.eval()
reid_model.to(device)


transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((128, 64)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


EMBEDDING_DIM = 2048


def extract_embedding(image):
    """
    Extracts a feature embedding from a cropped player image using the Re-ID model.
    """
    if image is None or image.shape[0] < 1 or image.shape[1] < 1:

        return np.zeros(EMBEDDING_DIM)
    try:

        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        with torch.no_grad():
            img_tensor = transform(rgb_image).unsqueeze(0).to(device)
            emb = reid_model(img_tensor).squeeze().cpu().numpy()
            return emb / np.linalg.norm(emb)
    except Exception as e:
        print(f"Error extracting embedding: {e}")
        return np.zeros(EMBEDDING_DIM)

ocr_reader = easyocr.Reader(['en'], gpu=False)


def extract_jersey_number(image):
    """
    Extracts jersey numbers from a cropped player image using EasyOCR.
    Adds basic pre-processing and confidence filtering for better accuracy.
    """
    if image is None or image.shape[0] < 1 or image.shape[1] < 1:
        return None
    try:

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        processed_image = clahe.apply(gray_image)

        result = ocr_reader.readtext(processed_image)

        best_number = None
        highest_conf = 0.0

        for bbox, text, conf in result:

            if text.isdigit() and 1 <= len(text) <= 2 and conf > 0.5:  # Tune confidence
                if conf > highest_conf:
                    highest_conf = conf
                    best_number = text
        return best_number
    except Exception as e:
        print(f"Error extracting jersey number: {e}")
        return None


# Dominant Color in HSV Histogram
def get_dominant_color(image):
    """
    Calculates a normalized HSV color histogram for a cropped image.
    Used as a color descriptor.
    """
    if image is None or image.size == 0 or image.shape[0] < 1 or image.shape[1] < 1:

        return np.zeros((8 * 8,), dtype=np.float32)
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        hist = cv2.calcHist([hsv], [0, 1], None, [8, 8], [0, 180, 0, 256])
        hist = cv2.normalize(hist, hist, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX).flatten()
        return hist
    except Exception as e:
        print(f"Error getting dominant color: {e}")
        return np.zeros((8 * 8,), dtype=np.float32)



GRID_SIZE = 9


def get_grid_position(center, frame_shape):
    """
    Calculates the grid cell a player's center falls into.
    Note: This is view-dependent. For true cross-camera spatial comparison,
    you need to project positions using homography.
    Args:
        center (tuple): (x, y) coordinates of the player's center.
        frame_shape (tuple): (height, width) of the frame.
    Returns:
        tuple: (gx, gy) grid coordinates.
    """
    h, w = frame_shape[0], frame_shape[1]

    gx = min(int(center[0] / (w / GRID_SIZE)), GRID_SIZE - 1) if w > 0 else 0
    gy = min(int(center[1] / (h / GRID_SIZE)), GRID_SIZE - 1) if h > 0 else 0
    return gx, gy


# ---------------------- Video Processing Function ----------------------

def process_video(video_path, tracker, video_name):
    """
    Processes a single video: performs detection, tracking, feature extraction,
    and stores track metadata.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video {video_path}")
        return {}, {}

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Output video setup
    output_video_path = f"output_{video_name}.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_video_path, fourcc, fps, (frame_width, frame_height))

    player_tracks = {}
    track_metadata = {}
    frame_idx = 0

    print(f"Starting processing for {video_name}...")
    while True:
        ret, frame = cap.read()
        if not ret:
            break


        results = model(frame)[0]
        detections = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])


            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)

            crop = frame[y1:y2, x1:x2]


            if crop.shape[0] > 0 and crop.shape[1] > 0:
                emb = extract_embedding(crop)
                detections.append(([x1, y1, x2 - x1, y2 - y1], conf, emb))

        # DeepSORT update
        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue

            tid = track.track_id
            bbox = track.to_ltwh()
            x, y, w, h = map(int, bbox)
            x2, y2 = x + w, y + h
            center = (x + w // 2, y + h // 2)

            grid_pos = get_grid_position(center, (frame_height, frame_width))


            x, y = max(0, x), max(0, y)
            x2, y2 = min(frame_width, x2), min(frame_height, y2)
            crop = frame[y:y2, x:x2]

            jersey = None
            if crop.shape[0] > 0 and crop.shape[1] > 0:
                jersey = extract_jersey_number(crop)

            color = get_dominant_color(crop)

            embedding_current = extract_embedding(crop)

            if tid not in player_tracks:
                player_tracks[tid] = {
                    "embeddings": [],
                    "frames": [],
                    "centers": [],
                    "jerseys": defaultdict(int),
                    "colors": []
                }

            player_tracks[tid]["embeddings"].append(embedding_current)
            player_tracks[tid]["frames"].append(frame_idx)
            player_tracks[tid]["centers"].append(center)
            if jersey:
                player_tracks[tid]["jerseys"][jersey] += 1
            player_tracks[tid]["colors"].append(color)

            # Draw bounding box and ID
            cv2.rectangle(frame, (x, y), (x2, y2), (0, 255, 0), 2)
            label = f"ID:{tid}"
            if jersey:
                label += f" | #{jersey}"
            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        out.write(frame)
        frame_idx += 1


    for tid in player_tracks:

        embeddings_list = player_tracks[tid]["embeddings"]
        avg_emb = np.mean(embeddings_list, axis=0) if embeddings_list else np.zeros(EMBEDDING_DIM)
        track_metadata[tid] = {
            "embedding": avg_emb / np.linalg.norm(avg_emb) if np.linalg.norm(avg_emb) > 0 else avg_emb,
            "frames": player_tracks[tid]["frames"],
            "centers": player_tracks[tid]["centers"],
            "jersey": max(player_tracks[tid]["jerseys"], key=player_tracks[tid]["jerseys"].get) if player_tracks[tid][
                "jerseys"] else None,
            "color": np.mean(player_tracks[tid]["colors"], axis=0) if player_tracks[tid]["colors"] else np.zeros((64,),
                                                                                                                 dtype=np.float32),

            "avg_grid": get_grid_position(np.mean(player_tracks[tid]["centers"], axis=0).astype(int),
                                          (frame_height, frame_width)) if player_tracks[tid]["centers"] else (0, 0)

        }

    cap.release()
    out.release()
    print(f"Finished processing {video_name}.")
    return player_tracks, track_metadata


def project_point(point, H_matrix):
    """
    Projects a 2D point using a homography matrix.
    Args:
        point (tuple or list): A 2-element tuple/list (x, y) representing the point.
        H_matrix (np.array): The 3x3 homography matrix.
    Returns:
        tuple: The projected 2D point (x', y').
    """

    point_homo = np.array([point[0], point[1], 1], dtype=np.float32)

    projected_point_homo = H_matrix @ point_homo

    if projected_point_homo[2] != 0:
        projected_point = (projected_point_homo[0] / projected_point_homo[2],
                           projected_point_homo[1] / projected_point_homo[2])
    else:

        print("Warning: Z-component of projected point is zero. Check homography.")
        projected_point = (point[0], point[1])
    return projected_point


# ---------------------- Main Logic ----------------------
if __name__ == '__main__':
    video1_path = "broadcast.mp4"
    video2_path = "tacticam.mp4"


    tracker1 = DeepSort(max_age=30, n_init=3)
    tracker2 = DeepSort(max_age=30, n_init=3)

    print("\nStarting video processing...")

    _, meta1 = process_video(video1_path, tracker1, "broadcast")
    _, meta2 = process_video(video2_path, tracker2, "tacticam")
    print("\nVideo processing complete. Starting player matching...")


    if not meta1:
        print("Warning: No player tracks found in Video 1 (broadcast.mp4). Check YOLO detections or video path.")
    if not meta2:
        print("Warning: No player tracks found in Video 2 (tacticam.mp4). Check YOLO detections or video path.")


    H = None
    mapping = []
    used_video2_ids = set()
    global_id_counter = 0

    sorted_meta1_items = sorted(meta1.items(), key=lambda item: len(item[1]["frames"]), reverse=True)

    print("\nAttempting to match players...")
    for id1, m1 in sorted_meta1_items:
        best_match_id2 = None
        highest_score = -1.0

        for id2, m2 in meta2.items():
            if id2 in used_video2_ids:
                continue


            overlap_frames = set(m1["frames"]) & set(m2["frames"])
            if len(overlap_frames) < 1:
                continue

            current_match_score = 0.0


            jersey_match_strength = 0.0
            if m1["jersey"] and m2["jersey"] and m1["jersey"] == m2["jersey"]:
                jersey_match_strength = 100.0
            current_match_score += jersey_match_strength


            embedding_similarity = 1 - cosine(m1["embedding"], m2["embedding"])
            current_match_score += embedding_similarity * 50.0

            spatial_proximity_score = 0.0
            if H is not None and m1["centers"] and m2["centers"]:

                avg_center1 = np.mean(m1["centers"], axis=0)
                projected_center1_to_v2 = project_point(avg_center1.astype(int), H)

                avg_center2 = np.mean(m2["centers"], axis=0)

                spatial_distance = np.linalg.norm(np.array(projected_center1_to_v2) - np.array(avg_center2))

                max_spatial_dist_consider = 200
                spatial_proximity_score = max(0, (
                            max_spatial_dist_consider - spatial_distance) / max_spatial_dist_consider) * 20.0

            else:
                if m1["avg_grid"] == m2["avg_grid"]:
                    spatial_proximity_score = 5.0

            current_match_score += spatial_proximity_score

            color_similarity = cv2.compareHist(m1["color"], m2["color"], cv2.HISTCMP_CORREL)
            current_match_score += color_similarity * 15.0

            if current_match_score > highest_score:
                highest_score = current_match_score
                best_match_id2 = id2


        MIN_CONFIDENT_MATCH_SCORE = 40.0

        if best_match_id2 is not None and highest_score >= MIN_CONFIDENT_MATCH_SCORE:
            print(
                f"-> Matched Global ID {global_id_counter}: Video1 ID {id1} and Video2 ID {best_match_id2} (Score: {highest_score:.2f})")
            mapping.append({
                "global_id": global_id_counter,
                "video1_id": id1,
                "video2_id": best_match_id2,

            })
            used_video2_ids.add(best_match_id2)
            global_id_counter += 1
        else:
            if best_match_id2 is None:
                print(
                    f"-> No suitable match found for Video1 ID {id1} (no candidates met temporal overlap or other filters).")
            else:
                print(
                    f"-> No confident match found for Video1 ID {id1}. Best score was {highest_score:.2f}, below threshold {MIN_CONFIDENT_MATCH_SCORE}.")

    # Save mapping to Excel
    df = pd.DataFrame(mapping)
    output_excel_path = "player_mapping.xlsx"
    df.to_excel(output_excel_path, index=False)

    print("\n✅ Processing complete!")
    print(f"Output videos saved to 'output_broadcast.mp4' and 'output_tacticam.mp4'.")
    print(f"Player mapping saved to '{output_excel_path}'.")
    if df.empty:
        print(
            "Note: The 'player_mapping.xlsx' is blank because no matches met the criteria. Check print statements for debugging scores.")
