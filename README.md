# Warehouse Lane Utilization and Box Tracking System

A comprehensive computer vision system for tracking boxes in warehouse staging areas, analyzing lane utilization, and generating detailed Excel reports for operational insights.

## 📋 Table of Contents
- [Overview](#overview)
- [Features](#features)
- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Usage](#usage)
- [Output Reports](#output-reports)
- [Lane Configuration](#lane-configuration)
- [Key Metrics](#key-metrics)
- [Troubleshooting](#troubleshooting)

## 🎯 Overview

This system uses computer vision and object tracking to monitor warehouse staging areas with multiple lanes. It tracks box movements, calculates lane utilization percentages, and generates comprehensive Excel reports at different time intervals (1-minute and 15-minute).

**Key Capabilities:**
- Real-time box detection and tracking
- 6-lane utilization analysis
- Entry/exit tracking for staging areas
- Automated Excel report generation
- Visual overlay on processed video

## ✨ Features

### Core Functionality
- **Box Detection & Tracking**: Uses YOLO-based object detection with tracking IDs
- **Lane-Based Analysis**: Monitors 6 configurable lanes with individual utilization metrics
- **Movement Detection**: Identifies stationary vs. moving boxes
- **ROI Management**: Tracks boxes entering and exiting staging areas
- **Direction Detection**: Determines if boxes enter from left, right, or are already in staging

### Reporting
- **Staging Time Reports**: Generated every 60 seconds
- **1-Minute Capacity Reports**: Lane utilization every minute
- **15-Minute Capacity Reports**: Aggregated analysis every 15 minutes
- **Detailed Lane Reports**: Per-lane performance metrics

### Metrics Tracked
- Box count per lane
- Utilization percentage per lane
- Utilized area vs. total area
- Inflow/outflow rates
- Average staging times
- Box dimensions and positions
- Entry direction tracking

## 💻 System Requirements

### Hardware
- GPU recommended (CUDA-compatible) for real-time processing
- Minimum 8GB RAM
- Sufficient storage for video files and reports

### Software
- Python 3.8+
- OpenCV 4.5+
- CUDA Toolkit (optional, for GPU acceleration)

## 🔧 Installation

### 1. Clone the Repository
```bash
git clone <repository-url>
cd warehouse-lane-tracking
```

### 2. Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Required Python Packages
```txt
opencv-python>=4.5.0
numpy>=1.19.0
pandas>=1.2.0
openpyxl>=3.0.0
ultralytics>=8.0.0  # For YOLO
torch>=1.9.0
torchvision>=0.10.0
```

## 📁 Project Structure

```
warehouse-lane-tracking/
│
├── config.py                 # Configuration and constants
├── lane_utils.py            # Lane calculation utilities
├── report_generator.py      # Excel report generation
├── box_tracker.py           # Box tracking class
├── main.py                  # Main processing script
├── requirements.txt         # Python dependencies
├── README.md               # This file
│
├── models/
│   └── best.pt             # Trained YOLO model weights
│
├── input/
│   └── [video files]       # Input videos
│
└── output/
    ├── videos/             # Processed videos with overlays
    ├── reports/            # Generated Excel reports
    └── logs/               # Processing logs
```

## ⚙️ Configuration

### config.py - Key Parameters

```python
# Video Settings
VIDEO_PATH = "path/to/your/video.mp4"
OUTPUT_VIDEO_PATH = "output_video.mp4"
MODEL_PATH = "path/to/yolo/weights.pt"
DURATION_SECONDS = 65 * 60  # Processing duration

# Report Generation Intervals
STAGING_EXCEL_INTERVAL = 60      # Staging report every 60 seconds
CAPACITY_1MIN_INTERVAL = 60      # 1-min capacity report
CAPACITY_EXCEL_INTERVAL = 900    # 15-min capacity report

# Tracking Parameters
HISTORY_LENGTH_SECONDS = 1.0     # Movement history duration
MOVEMENT_THRESHOLD = 5           # Pixels for stationary detection

# Lane Colors (BGR format)
LANE_COLORS = {
    1: (255, 0, 0),      # Red
    2: (0, 255, 0),      # Green
    3: (0, 0, 255),      # Blue
    4: (255, 255, 0),    # Cyan
    5: (255, 0, 255),    # Magenta
    6: (0, 255, 255)     # Yellow
}
```

## 🚀 Usage

### Basic Usage

```bash
python main.py
```

### With Custom Video

```python
from config import VIDEO_PATH, OUTPUT_VIDEO_PATH
from main import process_video

# Process video
process_video(
    video_path="path/to/video.mp4",
    output_path="output.mp4",
    model_path="models/best.pt"
)
```

### Processing Flow

1. **Video Loading**: System loads video and initializes lane polygons
2. **Frame Processing**: Each frame is analyzed for box detection
3. **Lane Assignment**: Boxes are assigned to lanes based on center point
4. **Tracking**: Box movements are tracked across frames
5. **Utilization Calculation**: Lane utilization computed in real-time
6. **Report Generation**: Excel files generated at specified intervals
7. **Video Output**: Processed video saved with visual overlays

## 📊 Output Reports

### 1. Staging Times Report (`box_staging_times_lanes_*.xlsx`)
Generated every 60 seconds

**Columns:**
- Box ID
- Lane
- In Time
- Out Time
- Total Staging Time (s)
- Box Width/Height/Area
- Entry Direction
- Status

### 2. 1-Minute Capacity Report (`lane_utilization_1min_*.xlsx`)
Generated every minute

**Columns:**
- Interval
- Start/End Time
- Total Inflow/Outflow
- Inflow/Outflow Rate
- Average Staging Time
- Lane 1-6 Utilization (%)
- Lane 1-6 Box Count
- Lane 1-6 Utilized Area
- Overall Avg Utilization
- Most/Least Utilized Lanes

### 3. 15-Minute Capacity Report (`lane_utilization_15min_*.xlsx`)
Generated every 15 minutes

**Same structure as 1-minute report with aggregated data**

### 4. Detailed Lane Report (`detailed_lane_utilization_*.xlsx`)
Optional detailed per-lane analysis

**Columns:**
- Interval
- Lane ID
- Utilization (%)
- Box Count
- Utilized/Total Area
- Efficiency Score

## 🗺️ Lane Configuration

### Lane Coordinate System
Lanes are defined using normalized coordinates (0-1 range) that scale to video dimensions.

**Format:**
```python
LANE_COORDINATES = {
    lane_id: [[x1, y1], [x2, y2], [x3, y3], ...]
}
```

### Current Configuration
The system supports 6 lanes with polygonal boundaries. Coordinates are automatically converted to pixel values based on video resolution.

**To Modify Lanes:**
1. Edit `LANE_COORDINATES` in `config.py`
2. Ensure coordinates are normalized (0.0 to 1.0)
3. Define polygons in clockwise or counter-clockwise order
4. Update `LANE_COLORS` if adding/removing lanes

## 📈 Key Metrics

### Lane Utilization Percentage
```
Utilization (%) = (Utilized Area / Total Lane Area) × 100
```

### Flow Rates
```
Inflow Rate = Total Inflow / Interval Duration (minutes)
Outflow Rate = Total Outflow / Interval Duration (minutes)
```

### Efficiency Score
```
Efficiency = Utilization (%) × Box Count / 100
```

### Box Assignment
- Boxes assigned to lanes based on **center point**
- If center point falls within lane polygon → box belongs to that lane
- Overlapping area calculated for accurate utilization metrics

## 🔍 Troubleshooting

### Common Issues

**1. No boxes detected**
- Check MODEL_PATH points to valid YOLO weights
- Verify video format is supported (MP4, AVI recommended)
- Ensure model is trained for box detection

**2. Incorrect lane assignments**
- Verify lane coordinates are correct
- Check coordinate normalization (should be 0-1)
- Use `draw_lanes_on_frame()` to visualize lane boundaries

**3. Reports not generating**
- Check write permissions in output directory
- Verify pandas and openpyxl are installed
- Check disk space availability

**4. Performance issues**
- Reduce video resolution
- Enable GPU acceleration (CUDA)
- Increase frame skip rate
- Process shorter video segments

**5. Tracking ID issues**
- Adjust MOVEMENT_THRESHOLD if boxes incorrectly marked stationary
- Modify HISTORY_LENGTH_SECONDS for tracking sensitivity
- Check occlusion handling in tracker

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### Performance Optimization

**For faster processing:**
```python
# In config.py
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame
RESIZE_FACTOR = 0.5         # Resize video to 50%
```

## 📝 Customization

### Adding New Lanes
1. Add lane coordinates to `LANE_COORDINATES`
2. Add color to `LANE_COLORS`
3. Update report generation to include new lane

### Custom Metrics
Extend `calculate_interval_metrics()` in `report_generator.py`:
```python
def calculate_custom_metric(lane_data):
    # Your custom calculation
    return metric_value
```

### Modified Report Format
Edit report generation functions in `report_generator.py`:
```python
def generate_custom_report(data):
    # Define custom columns
    df = pd.DataFrame(data)
    df.to_excel('custom_report.xlsx')
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Commit changes with clear messages
4. Submit pull request with description



**Last Updated:** November 2024  
**Version:** 1.0.0
