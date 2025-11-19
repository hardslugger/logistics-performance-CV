import cv2
import numpy as np
from collections import deque
from ultralytics import YOLO
import torch
from datetime import datetime, timedelta

import config
import utils
import reporting

# Check GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"🚀 Using device: {device.upper()}")
if device == 'cuda':
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA Version: {torch.version.cuda}")
    print(f"   Memory Available: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

print("🔄 Loading YOLO model...")
model = YOLO(config.MODEL_PATH)
model.to(device)  # Move model to GPU
model.model.names.update({0: "box", 1: "forklift"})
print("✅ YOLO model loaded successfully")

# Video setup
cap = cv2.VideoCapture(config.VIDEO_PATH)

# Video properties
fps = cap.get(cv2.CAP_PROP_FPS)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(config.OUTPUT_VIDEO_PATH, fourcc, fps, (width, height))

# Duration and frame control
duration_seconds = 65 * 60  # Example: 65 minutes
max_frames = int(fps * duration_seconds)
frame_count = 0

# --- New Polygonal Lane and ROI Initialization ---
print("📐 Initializing polygonal lanes based on normalized coordinates...")
all_lane_points = []
for i, lane_str in enumerate(config.LANE_COORDINATES_NORMALIZED):
    # Parse string of numbers
    coords = np.array(lane_str.split(), dtype=float).reshape(-1, 2)
    # Scale normalized coordinates to frame dimensions
    coords[:, 0] *= width
    coords[:, 1] *= height
    # Create integer polygon for OpenCV
    polygon = coords.astype(np.int32)
    config.LANE_POLYGONS.append(polygon)
    
    # Calculate and store the area of the polygon
    lane_area = cv2.contourArea(polygon)
    config.LANE_TOTAL_AREAS.append(lane_area)
    
    # Collect all points to calculate an overall bounding box
    all_lane_points.extend(polygon)
    print(f"   Lane {i+1}: Area = {lane_area:,.1f} pixels², {len(polygon)} vertices")

# Define the overall ROI as the bounding box encompassing all defined lanes
if all_lane_points:
    all_lane_points = np.array(all_lane_points)
    x, y, w, h = cv2.boundingRect(all_lane_points)
    roi_left, roi_top, roi_right, roi_bottom = x, y, x + w, y + h
    config.ROI_POLYGON = np.array([
        [roi_left, roi_top], [roi_right, roi_top],
        [roi_right, roi_bottom], [roi_left, roi_bottom]
    ], dtype=np.int32)
else: # Fallback if no lanes are defined
    roi_left, roi_top, roi_right, roi_bottom = 0, 0, width, height
    config.ROI_POLYGON = np.array([[0,0], [width,0], [width,height], [0,height]], dtype=np.int32)


# The total area of the staging zone is the sum of all individual lane areas
config.ROI_TOTAL_AREA = sum(config.LANE_TOTAL_AREAS)

print(f"📊 Total Staging Area (sum of all lanes): {config.ROI_TOTAL_AREA:,.1f} pixels²")
print(f"📍 Overall ROI Bounding Box: ({roi_left},{roi_top}) to ({roi_right},{roi_bottom})")

# Pre-calculate the horizontal (x-axis) bounds of each lane
lane_x_bounds = []
for polygon in config.LANE_POLYGONS:
    x_min = np.min(polygon[:, 0])
    x_max = np.max(polygon[:, 0])
    lane_x_bounds.append((x_min, x_max))


# Excel generation timing
last_staging_excel_generation = 0
last_capacity_excel_generation = 0

# Tracking data for 15-minute intervals
interval_data = [] # Stores calculated metrics for each completed interval
current_interval_start = 0
current_interval_capacity_data = [] # Stores raw utilization data for current interval
current_interval_inflow = 0  # boxes that entered ROI in current interval
current_interval_outflow = 0  # boxes that exited ROI in current interval

# Tracking history and threshold
track_history = {}
history_length = int(fps * config.HISTORY_LENGTH_SECONDS)
movement_threshold = config.MOVEMENT_THRESHOLD_PIXELS

# Time tracking dictionaries
start_frames = {}  # {track_id: frame_number} - when box became stationary in ROI
end_frames = {}    # {track_id: frame_number} - when box left ROI
box_roi_status = {}  # {track_id: True/False} - current ROI status for each box
boxes_exited_roi = {}  # {track_id: True} - boxes that have exited ROI

# Track box dimensions for area calculation
box_dimensions = {}  # {track_id: (width, height)} - store box dimensions

# Track misplaced boxes (outside staging area)
misplaced_boxes = {}  # {track_id: frame_number} - when box was first detected outside ROI
all_detected_boxes = set()  # Track all boxes that have been detected

# Track entry direction and boxes already in staging area
box_entry_direction = {}  # {track_id: 'left'/'right'/'already_in_staging'} - direction from which box entered
box_first_position = {}  # {track_id: x_coordinate} - first detected x position
boxes_already_in_staging = set()  # Track boxes that were already in staging area when first detected

# Track last seen frame for all boxes (for misplaced duration)
box_last_seen_frame = {}  # {track_id: frame_count}

# Track exit direction
box_exit_direction = {} # {track_id: 'left'/'right'}

# Process frames
while cap.isOpened() and frame_count < max_frames:
    success, frame = cap.read()
    if not success:
        break

    result = model.track(frame, persist=True, tracker="bytetrack.yaml")[0]

    # --- Frame-specific data initialization ---
    current_boxes_in_roi = []
    current_stationary_in_roi = []
    current_boxes_in_roi_data = []
    
    # NEW: Dictionary to store the track_id and x1 coordinate of boxes in each lane
    lane_boxes_data = {i: [] for i in range(config.NUM_LANES)}

    class_ids = []
    if result.boxes and result.boxes.id is not None:
        boxes = result.boxes.xyxy.cpu()
        track_ids = result.boxes.id.int().cpu().tolist()
        class_ids = result.boxes.cls.int().cpu().tolist()

        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            if class_id == 0:  # Only process boxes
                box_last_seen_frame[track_id] = frame_count # Update last seen frame for all boxes
                x1, y1, x2, y2 = map(int, box)
                center = ((x1 + x2) // 2, (y1 + y2) // 2)
                
                # Store box dimensions
                box_width = x2 - x1
                box_height = y2 - y1
                box_area = box_width * box_height
                box_dimensions[track_id] = (box_width, box_height)
                all_detected_boxes.add(track_id)

                # Check if box centroid is inside the overall ROI bounding box
                is_in_roi = cv2.pointPolygonTest(config.ROI_POLYGON, center, False) >= 0
                entry_direction = utils.determine_entry_direction(track_id, center[0], is_in_roi,
                                                                  box_first_position, box_entry_direction,
                                                                  boxes_already_in_staging, frame_count, width, fps)

                if not is_in_roi and track_id not in misplaced_boxes and \
                   track_id not in start_frames and track_id not in boxes_already_in_staging:
                    misplaced_boxes[track_id] = frame_count
                    print(f"⚠️  Box ID {track_id} detected as MISPLACED at frame {frame_count} ({frame_count/fps:.1f}s), Entry: {entry_direction}")

                if track_id in box_roi_status and box_roi_status[track_id] and not is_in_roi:
                    if track_id in start_frames and track_id not in end_frames:
                        end_frames[track_id] = frame_count
                        boxes_exited_roi[track_id] = True
                        current_interval_outflow += 1
                        elapsed_time = (frame_count - start_frames[track_id]) / fps
                        
                        # Determine and store exit direction
                        exit_direction = utils.determine_exit_direction(center[0], width)
                        box_exit_direction[track_id] = exit_direction
                        
                        print(f"🚪 Box ID {track_id} LEFT ROI towards {exit_direction} at frame {frame_count} ({frame_count/fps:.1f}s) - Staging time: {elapsed_time:.1f}s, Area: {box_area}px², Entry: {entry_direction}")
                
                box_roi_status[track_id] = is_in_roi

                if is_in_roi:
                    current_boxes_in_roi.append(track_id)
                    current_boxes_in_roi_data.append((track_id, (x1, y1, x2, y2)))

                    # Assign box to a polygonal lane
                    lane_idx = utils.get_lane_for_box(center, config.LANE_POLYGONS)
                    if lane_idx != -1:
                        # NEW: Collect track_id and x1 for filtering later
                        lane_boxes_data[lane_idx].append((track_id, x1))

                    if track_id not in track_history:
                        track_history[track_id] = deque(maxlen=history_length)
                    track_history[track_id].append(center)

                    if len(track_history[track_id]) >= history_length:
                        movements = [np.linalg.norm(np.array(track_history[track_id][i]) - np.array(track_history[track_id][i - 1])) for i in range(1, len(track_history[track_id]))]
                        if all(m < movement_threshold for m in movements):
                            current_stationary_in_roi.append(track_id)
                            if track_id not in start_frames:
                                start_frames[track_id] = frame_count
                                if track_id not in boxes_already_in_staging:
                                    current_interval_inflow += 1
                                print(f"📦 Box ID {track_id} became stationary in ROI at frame {frame_count} ({frame_count/fps:.1f}s), Area: {box_area}px², Entry: {entry_direction}")

        # --- Drawing Logic ---
        for box, track_id, class_id in zip(boxes, track_ids, class_ids):
            x1, y1, x2, y2 = map(int, box)
            current_color = config.CLASS_COLORS[class_id]
            if class_id == 0:
                if track_id in boxes_exited_roi: current_color = config.EXITED_COLOR
                elif track_id in boxes_already_in_staging and track_id in current_stationary_in_roi: current_color = config.ALREADY_IN_STAGING_COLOR
                elif track_id in current_stationary_in_roi: current_color = config.STATIONARY_COLOR
                elif track_id in current_boxes_in_roi and track_id in start_frames: current_color = config.MOVING_IN_ROI_COLOR
                elif track_id in misplaced_boxes: current_color = config.MISPLACED_COLOR
            cv2.rectangle(frame, (x1, y1), (x2, y2), current_color, 2)
            class_name = "box" if class_id == 0 else "forklift"
            entry_dir_short = {"left": "(L)", "right": "(R)", "already_in_staging": "(already)"}.get(box_entry_direction.get(track_id), '')
            label = f'{class_name} ID:{track_id} {entry_dir_short}'
            if class_id == 0:
                if track_id in boxes_exited_roi: label += " (EXITED)"
                elif track_id in misplaced_boxes and track_id not in start_frames: label += " (MISPLACED)"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)
            if class_id == 0 and track_id in start_frames and track_id not in boxes_exited_roi:
                elapsed_seconds = (frame_count - start_frames[track_id]) / fps
                minutes, seconds = divmod(int(elapsed_seconds), 60)
                timer_text = f"Time: {minutes:02}:{seconds:02}"
                cv2.putText(frame, timer_text, (x1, y1 - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, current_color, 2)

    # Calculate overall ROI area utilization (using the bounding box)
    area_utilization_percent, total_utilized_area = utils.calculate_area_utilization(
        current_boxes_in_roi_data, (roi_left, roi_top, roi_right, roi_bottom), config.ROI_TOTAL_AREA)

    # NEW: Calculate lane utilization based on Right-to-Left occupancy of STATIONARY boxes only
    current_lane_utilization_percentages = []
    for i in range(config.NUM_LANES):
        lane_x_min, lane_x_max = lane_x_bounds[i]
        lane_width = lane_x_max - lane_x_min
        
        # Get all boxes' data for the current lane
        all_boxes_in_lane = lane_boxes_data[i]
        
        # Filter to get only the x1 coordinates of stationary boxes in this lane
        stationary_boxes_x1 = [x1 for track_id, x1 in all_boxes_in_lane if track_id in current_stationary_in_roi]
        
        util_percent = 0.0
        if stationary_boxes_x1 and lane_width > 0:
            # Find the leftmost point of any STATIONARY box in the lane
            occupied_front_x = min(stationary_boxes_x1)
            # The utilized part is from the leftmost stationary box to the right edge of the lane
            occupied_width = lane_x_max - occupied_front_x
            util_percent = (occupied_width / lane_width) * 100
            # Cap the value between 0 and 100 for safety
            util_percent = max(0, min(100, util_percent))
            
        current_lane_utilization_percentages.append(util_percent)


    # Draw the polygonal lanes
    for polygon in config.LANE_POLYGONS:
        cv2.polylines(frame, [polygon], isClosed=True, color=config.LANE_COLOR, thickness=2)
    
    # Draw the overall ROI bounding box
    cv2.polylines(frame, [config.ROI_POLYGON], isClosed=True, color=config.ROI_COLOR, thickness=3)

    # Display info panel
    info_x, info_y = 10, 10
    panel_height = 200 + config.NUM_LANES * 20
    cv2.rectangle(frame, (info_x, info_y), (info_x + 500, info_y + panel_height), (0, 0, 0), -1)
    cv2.rectangle(frame, (info_x, info_y), (info_x + 500, info_y + panel_height), (255, 255, 255), 2)
    total_boxes = sum(1 for cls in class_ids if cls == 0)
    total_forklifts = sum(1 for cls in class_ids if cls == 1)
    tracked_boxes_in_roi = len([tid for tid in current_boxes_in_roi if tid in start_frames])
    current_misplaced = len([tid for tid in misplaced_boxes if tid not in start_frames])
    cv2.putText(frame, f'Total Boxes: {total_boxes}', (info_x + 10, info_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Total Forklifts: {total_forklifts}', (info_x + 10, info_y + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Stationary in ROI: {len(current_stationary_in_roi)}', (info_x + 10, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Tracked in ROI: {tracked_boxes_in_roi}', (info_x + 10, info_y + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Exited ROI: {len(boxes_exited_roi)}', (info_x + 10, info_y + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    cv2.putText(frame, f'Misplaced Boxes: {current_misplaced}', (info_x + 10, info_y + 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, config.MISPLACED_COLOR, 2)
    # Note: 'Overall Area Util' is still based on total box area for a general overview
    cv2.putText(frame, f'Overall Area Util (Approx): {area_utilization_percent:.1f}%', (info_x + 10, info_y + 140), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    y_offset_lanes = info_y + 160
    for i, util_percent in enumerate(current_lane_utilization_percentages):
        # NEW: Updated label for new utilization logic
        cv2.putText(frame, f'Lane {i+1} Static Util (R-L): {util_percent:.1f}%', (info_x + 10, y_offset_lanes + i * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 255, 100), 2)

    # --- Excel Reporting Logic (Unchanged) ---
    current_time_seconds = frame_count / fps
    if current_time_seconds - last_staging_excel_generation >= config.STAGING_EXCEL_INTERVAL_SECONDS:
        reporting.generate_staging_excel(start_frames, end_frames, misplaced_boxes, current_time_seconds, fps,
                                         box_dimensions, box_entry_direction, boxes_already_in_staging,
                                         box_last_seen_frame, box_exit_direction)
        last_staging_excel_generation = current_time_seconds
    
    current_interval_capacity_data.append({
        'timestamp': current_time_seconds,
        'overall_area_utilization': area_utilization_percent,
        'overall_utilized_area': total_utilized_area,
        'lane_utilization': current_lane_utilization_percentages
    })

    if current_time_seconds - last_capacity_excel_generation >= config.CAPACITY_EXCEL_INTERVAL_SECONDS:
        interval_metrics = reporting.calculate_interval_metrics(
            current_interval_capacity_data, current_interval_inflow, current_interval_outflow,
            start_frames, end_frames, current_interval_start, current_time_seconds, fps,
            config.LANE_POLYGONS, config.LANE_TOTAL_AREAS # Pass new lane data
        )
        interval_data.append({'start_time': current_interval_start, 'end_time': current_time_seconds, **interval_metrics})
        reporting.generate_capacity_excel(interval_data, current_time_seconds)
        last_capacity_excel_generation = current_time_seconds
        current_interval_start = current_time_seconds
        current_interval_capacity_data = []
        current_interval_inflow = 0
        current_interval_outflow = 0

    out.write(frame)
    frame_count += 1

# --- Finalization (Unchanged, but uses updated data structures) ---
final_time = frame_count / fps
if current_interval_capacity_data:
    final_interval_metrics = reporting.calculate_interval_metrics(
        current_interval_capacity_data, current_interval_inflow, current_interval_outflow,
        start_frames, end_frames, current_interval_start, final_time, fps,
        config.LANE_POLYGONS, config.LANE_TOTAL_AREAS
    )
    interval_data.append({'start_time': current_interval_start, 'end_time': final_time, **final_interval_metrics})

reporting.generate_excel_files(start_frames, end_frames, misplaced_boxes, interval_data, final_time, fps,
                               box_dimensions, box_entry_direction, boxes_already_in_staging,
                               config.LANE_POLYGONS, config.LANE_TOTAL_AREAS,
                               box_last_seen_frame, box_exit_direction)

cap.release()
out.release()
print(f"\n🎉 Video processing complete!")
# Final summary prints...