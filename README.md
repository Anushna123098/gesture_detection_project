# 🖐️ Gesture Detection using MediaPipe & Deep Learning

This project is a real-time **gesture detection system** that recognizes hand gestures using **MediaPipe**, processes them using a trained **neural network**, and performs system-level actions like mute/unmute or volume control.

## 🚀 Features

- 🎥 Real-time webcam gesture detection
- 📊 Uses hand landmark data via **MediaPipe**
- 🧠 Trained using **TensorFlow/Keras**
- 💾 Gesture dataset saved in CSV format
- 💻 Automates keyboard actions using **PyAutoGUI**
- 👋 Supports multiple gestures:
  - Mute
  - Unmute
  - Increase Volume
  - Decrease Volume

## 🛠️ Technologies Used

- Python 3
- MediaPipe
- OpenCV
- TensorFlow / Keras
- NumPy
- Pandas
- PyAutoGUI

## 📁 Project Structure

```bash
gesture_detection_project/
├── dataset/                      # Raw image data (if any)
├── landmark_data/               # CSV files with gesture landmark data
├── venv/                        # Python virtual environment
├── modelss.py                   # Model training with raw image data
├── train_landmark_model.py      # Training using landmark CSV
├── detect_gesture.py            # Real-time detection with image model
├── detect_landmark_gesture.py   # Real-time detection using landmark model
├── collect_landmark_data.py     # Script to collect gesture landmarks
├── merge_landmark_data.py       # Combine multiple CSV files
├── recognise_gesture.py         # Recognize gesture and take action
├── requirements.txt             # Python dependencies
└── README.md                    # Project info (this file)

▶️ How to Run
1. Clone the repo
git clone https://github.com/Anushna123098/gesture_detection_project.git
cd gesture_detection_project
2. Create & activate virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
3. Install dependencies
pip install -r requirements.txt
4. Collect Gesture Data (Optional)
python collect_landmark_data.py
5. Train the Model
python train_landmark_model.py
6. Detect Gesture in Real-Time
python detect_landmark_gesture.py
During detection, the system will respond with volume/mute changes using pyautogui.
