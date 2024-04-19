# opencv-realtime-tracking

This project implements real-time tracking using OpenCV in Python. It provides functionalities such as video processing with a tracking algorithm, dynamic ROI selection, and alerts through sound notifications when abnormal movements are detected.

## Features

- Video processing with support for OpenCV tracking algorithms like CSRT and KCF.
- Dynamic Region of Interest (ROI) selection during video playback.
- Sound notifications for detected anomalies based on movement thresholds.
- Adjustable video window and processing settings.

## Requirements

- Python 3.7+
- OpenCV
- Click
- NumPy
- beepy (for sound notifications)

## Installation

To set up your environment to run this code, follow these steps:

1. Clone the repository to your local machine.

   ```bash
   git clone <repository-url>
   cd opencv-realtime-tracking

python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`

pip install -r requirements.txt

## Usage 
```bash
python -m src process-video --help # display options 
python -m src process-video --path2video /path/to/video --winsize 800 800 
```