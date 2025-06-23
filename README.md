# Football Player Multi-Camera Tracking

This project addresses the challenge of identifying and tracking individual football players across multiple video camera angles (Assignment 1).

## Table of Contents

1. [Approach](#1-approach)
2. [Pipeline](#2-pipeline)
3. [Models Used](#3-models-used)
4. [Explanation Flow](#4-explanation-flow)
5. [Setup Instructions (Local Machine)](#5-setup-instructions-local-machine)
6. [Setup Instructions (Kaggle)](#6-setup-instructions-kaggle)
7. [Output](#7-output)

---

## 1. Approach

The core approach is a hybrid method that combines robust object detection with a multi-object tracking framework, enhanced by various features for re-identification:

### Detection:

* **YOLOv8** is used to accurately detect players in each frame.

### Tracking & Re-identification:

* **DeepSORT** is employed to track players over time using a combination of motion prediction (Kalman filter) and appearance features (embeddings).

### Feature Extraction:

* **Appearance Embedding:** Deep feature vector from a pre-trained ResNet-50 model.
* **Jersey Number:** Extracted using EasyOCR.
* **Dominant Jersey Color:** HSV histogram to capture the jersey color.
* **Grid Position:** Coarse spatial location on the field.

### Cross-Camera Matching:

After tracking players independently in each video, a matching algorithm compares players across videos using a weighted scoring system based on:

* Exact jersey number match (highest weight)
* Appearance embedding similarity (cosine distance)
* Dominant color similarity (medium weight)
* Spatial proximity using grid (low weight)
* Temporal overlap: Players must appear in both videos at roughly the same time.

---

## 2. Pipeline

### Initialization:

* Determine the computing device (GPU or CPU).
* Load the YOLOv8 model (`best.pt`).
* Load the modified ResNet-50 model.
* Initialize EasyOCR reader.
* Initialize DeepSORT trackers.

### Video Processing (`process_video` function):

* Read frames from a video.
* For each frame:

  * Detect players using YOLO.
  * For each detection:

    * Extract appearance embeddings using ResNet-50.
    * Extract jersey number using EasyOCR.
    * Calculate dominant jersey color histogram.
    * Determine grid position.
    * Store metadata.
  * Update DeepSORT tracker.
  * Draw bounding boxes and IDs on output frame.
* After video ends:

  * Aggregate features (average embedding, most frequent jersey number, color, etc.).

### Cross-Video Matching (Assignment 1):

* Call `process_video` for both `broadcast.mp4` and `tacticam.mp4`.
* For each track in broadcast:

  * Compare with unmatched tracks in tacticam.
  * Filter candidates based on temporal overlap.
  * Compute a weighted score for each pair using:

    * Jersey match
    * Cosine similarity
    * Grid position match
    * Dominant color similarity
  * Match the best scoring pair (if above threshold).
* Save matched pairs to `player_mapping.xlsx`.

---

## 3. Models Used

### YOLOv8 (Ultralytics)

* Used for player detection.
* Custom-trained weights: `best.pt`

### ResNet-50 (Torchvision)

* Used for extracting appearance embeddings.
* Final layer replaced with `torch.nn.Identity()`.

### EasyOCR

* Used for recognizing jersey numbers.
* Language: English (`en`)

### DeepSORT (deep\_sort\_realtime)

* Used for tracking and maintaining consistent IDs.
* Combines Kalman filtering and appearance-based matching.

---

## 4. Explanation Flow

1. **Setup:** Import libraries, detect device (CUDA/CPU), load models.
2. **Detection + Tracking:**

   * For each video, process every frame to detect and track players.
   * Extract features and maintain track-specific metadata.
3. **Post Processing:**

   * Aggregate metadata across time.
   * Match players across videos using similarity metrics.
4. **Output:**

   * Annotated videos and a player mapping Excel sheet.

---

## 5. Setup Instructions (Local Machine)
```bash
model weights : https://drive.google.com/file/d/1-5fOSHOSB9UXyP_enOoZNAMScrePVcMD/view
```
### Step 1: Clone the Repository

```bash
git clone <https://github.com/pranavrw/Mapper>
cd <Mapper>
```

### Step 2: Create Virtual Environment

```bash
python -m venv venv
# Activate:
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install opencv-python-headless torch torchvision ultralytics easyocr deep-sort-realtime numpy pandas scipy
```

### Step 4: Add Data & Run

* Place `broadcast.mp4`, `tacticam.mp4`, and `best.pt` in the working directory.

```bash
python main.py
```

---

## 6. Setup Instructions (Kaggle)

### Step 1: Create a New Notebook

* Go to Kaggle → Notebooks → New Notebook.

### Step 2: Enable GPU

* In **Settings**, enable GPU (P100 or T4).
* Ensure Internet is ON.

### Step 3: Upload Dataset

* Create and upload a dataset with `best.pt`, `broadcast.mp4`, and `tacticam.mp4`.
* Attach it to the notebook via **+ Add Data**.

### Step 4: Install Dependencies

```python
!pip install opencv-python-headless torch torchvision ultralytics easyocr deep-sort-realtime pandas scipy
```

### Step 5: Adjust Paths

```python
video1_path = "/kaggle/input/your-dataset-name/broadcast.mp4"
video2_path = "/kaggle/input/your-dataset-name/tacticam.mp4"
model_path = "/kaggle/input/your-dataset-name/best.pt"
```

### Step 6: Run and Download Output

* Output files will be saved in `/kaggle/working/`:

  * `output_broadcast.mp4`
  * `output_tacticam.mp4`
  * `player_mapping.xlsx`

---

## 7. Output

After execution, the following files are generated:

* `output_broadcast.mp4`: Annotated video with tracking IDs from `broadcast.mp4`
* `output_tacticam.mp4`: Annotated video with tracking IDs from `tacticam.mp4`
* `player_mapping.xlsx`: Excel file mapping players across both videos with:

  * `global_id`: Unified player ID
  * `video1_id`: DeepSORT ID from broadcast view
  * `video2_id`: DeepSORT ID from tacticam view

Review videos visually and verify mapping using the Excel file.

---
