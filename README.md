# Project Overview

This project tries to determine country where the video was recorded, by analyzing the driving side and finding the language of the text in the video frames. It uses computer vision techniques to detect the driving side and employs OCR (Optical Character Recognition) to identify languages from text in images.

## Features
1. **Driving Side Detection**: Determines whether the driving side is left or right based on video input.
2. **Map Marking**: Marks multiple countries on a map and calculates distances from a given location.
3. **Language Detection**: Detects languages from text in image frames using OCR.

---

## Requirements

### Prerequisites
- Python 3.8 or higher
- Pip (Python package manager)

### Libraries
Install the required Python libraries using the following command:

```bash
pip install -r requirements.txt
```


## Instructions to Run

### On Windows
1. **Clone the Repository**:
   ```cmd
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up Virtual Environment**:
   ```cmd
   python -m venv venv
   venv\Scripts\activate
   ```

3. **Install Dependencies**:
   ```cmd
   pip install -r requirements.txt
   ```
   
4. **Run the Program**:
   ```cmd
   python geo_loc.py <path-to-video> <actual-coordinates>
   ```

---

### On Linux
1. **Clone the Repository**:
   ```bash
   git clone <repository-url>
   cd <repository-folder>
   ```

2. **Set Up Virtual Environment**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
   
4. **Run the Program**:
   ```bash
   python geo_loc.py <path-to-video> <actual-coordinates>
   ```

---

## Actual Coordinates are needed to calculate distances from the given location. The coordinates should be in the format `latitude,longitude` (e.g., `35.6895,139.6917` for Tokyo, Japan).

---

## Test Datasets

- [Japan](https://www.kaggle.com/datasets/ashikadnan/driving-video-for-lane-detection-various-weather) coordinates: `35.6895, 139.6917`
- [China](https://www.scidb.cn/en/detail?dataSetId=804399692560465920) coordinates: `39.9042, 116.4074`
- [Switzerland Zurich](https://rpg.ifi.uzh.ch/event_driving_datasets.html) coordinates: `47.3769, 8.5417`
- [Taiwan](https://aliensunmin.github.io/project/dashcam/) coordinates: `25.0330, 121.5654`

---

## Notes
- Ensure that the required input files (e.g., video files, image frames) are placed in the appropriate directories before running the scripts.
- For GPU support in EasyOCR, ensure that the necessary GPU drivers and libraries (e.g., CUDA) are installed.

---

## Output
- **Map Marking**: Generates an HTML file (`map_with_multiple_distances.html`) in the `output` directory.

---

## Troubleshooting
- If GPU support fails for EasyOCR, the script will automatically fall back to CPU mode.
- Ensure that the `output` directory exists for saving map files. Create it if it does not exist:
  ```bash
  mkdir output
  ```
