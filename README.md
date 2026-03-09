# Football Ball Possession Estimation using Computer Vision

A computer vision pipeline that automatically estimates team ball possession statistics from football broadcast videos using deep learning and color-based team identification.

The system detects players, referees, and the ball using YOLO object detection models and determines which team controls the ball by identifying the nearest player and classifying their team via jersey color segmentation.

# Project Overview

Football analytics often requires manual annotation of ball possession, which is time-consuming and subjective.

This project proposes an automated pipeline that:

1. Detects players, referees, and the ball using YOLO object detection models
2. Identifies teams using HSV-based jersey color detection
3. Assigns ball possession based on nearest player centroid
4. Aggregates frame-by-frame possession statistics

The final output includes ball possession percentages for both teams over a match segment.

# Pipeline

The complete processing pipeline is shown below:


<img width="158" height="627" alt="github1 drawio" src="https://github.com/user-attachments/assets/10202a61-ad65-470b-87dd-bf5f8c939158" />

# Dataset
The project uses a combination of datasets:
### SoccerNet Tracking Dataset
Used for player and ball annotations in football broadcast videos.
### COCO 2017 Dataset
Used to augment ball detection training data using the sports ball class.

Datasets were transformed into YOLO format before training.

# Model Training

Two object detection models were trained and compared:

| Model   | Description                                             |
| ------- | ------------------------------------------------------- |
| YOLOv8m | Baseline object detection model                         |
| YOLOv9m | Improved architecture with enhanced feature aggregation |

### Training Configuration

| Parameter             | Value                     |
| --------------------- | ------------------------- |
| Epochs                | 300                       |
| Image Size            | 1056                      |
| Optimizer             | Adam                      |
| Hyperparameter Search | Bayesian Optimization     |
| Augmentation          | Mosaic, flipping, scaling |

# Model Evaluation

Models were evaluated using standard object detection metrics:

* **Precision**
* **Recall**
* **mAP50**
* **mAP50–95**

Evaluation was performed on the SoccerNet validation dataset.

The best-performing model was used for the possession estimation pipeline.

# Team Identification

Teams are identified using jersey color segmentation in HSV color space.

Steps:

1. Extract player bounding box
2. Crop jersey region
3. Convert to HSV color space
4. Apply predefined color masks
5. Assign team label

This approach works well for broadcast footage with distinct jersey colors.

# Ball Possession Estimation

Ball possession is estimated by assigning the ball to the nearest detected player.

For each frame:

1. Compute centroid of detected ball
2. Compute centroids of all detected players
3. Calculate Euclidean distance
4. Assign ball to closest player
5. Determine player's team
6. Update possession counter

Example:

```
Ball → nearest player → team label → possession count
```

Final possession is calculated as:

```
Team Possession (%) = (Team Frames / Total Frames) × 100
```

# Project Structure

```
football-ball-possession-estimator
│
├── README.md
├── requirements.txt
│
├── dataset/
│
├── src/
│   ├── collect_ball_dataset.py
│   ├── transform_soccerNet_labels.py
│   ├── train_yolo_models.py
│   ├── evaluate_models.py
│   ├── team_color_detection.py
│   └── possession_estimation.py
│
├── notebooks/
│   └── experiments.ipynb
│
├── demo/
│   ├── possession_demo.gif
│   └── sample_video.mp4
│
└── results/
```

# Installation

Clone the repository:

```
git clone https://github.com/fareasy/football-possession-estimator.git
cd football-ball-possession-estimator
```

Install dependencies:

```
pip install -r requirements.txt
```

# Usage

Run possession estimation on a video:

```
python src/possession_estimation.py --video demo/sample_video.mp4
```

Output example:

```
Team A Possession: 63%
Team B Possession: 37%
```

# 🔧 Technologies Used

* Python
* PyTorch
* YOLOv8 / YOLOv9
* OpenCV
* NumPy
* FiftyOne
* SAHI

# Key Features

* Automatic player and ball detection
* Team classification via jersey color
* Frame-by-frame ball possession estimation
