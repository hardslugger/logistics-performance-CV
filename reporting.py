import pandas as pd
from datetime import datetime, timedelta
import config

def calculate_interval_metrics(interval_capacity_data, interval_inflow, interval_outflow,
                               start_frames, end_frames, interval_start_time, interval_end_time, fps,
                               lane_boundaries, lane_total_areas):
    """Calculate metrics for a 15-minute interval, including lane-wise utilization."""

    # Average overall area utilization
    if interval_capacity_data:
        avg_area_utilization = sum(entry['overall_area_utilization'] for entry in interval_capacity_data) / len(interval_capacity_data)
        avg_utilized_area = sum(entry['overall_utilized_area'] for entry in interval_capacity_data) / len(interval_capacity_data)
    else:
        avg_area_utilization = 0
        avg_utilized_area = 0

    # Calculate average lane utilization
    avg_lane_utilizations = {}
    for i in range(config.NUM_LANES):
        lane_data_points = [entry['lane_utilization'][i] for entry in interval_capacity_data if i < len(entry.get('lane_utilization', []))]
        avg_lane_utilizations[f'lane_{i+1}_avg_util'] = sum(lane_data_points) / len(lane_data_points) if lane_data_points else 0

    # Calculate average staging time for boxes that completed staging in this interval
    staging_times = []
    # Filter for boxes that *started* in this interval
    boxes_started_in_interval = {box_id: start_frame for box_id, start_frame in start_frames.items()
                                 if (start_frame / fps) >= interval_start_time and (start_frame / fps) < interval_end_time}

    # Add staging times for boxes that exited in this interval (and started within or before)
    for box_id, end_frame in end_frames.items():
        end_time_s = end_frame / fps
        if interval_start_time <= end_time_s <= interval_end_time and box_id in start_frames:
            staging_time_s = (end_frame - start_frames[box_id]) / fps
            staging_times.append(staging_time_s)

    # Also include ongoing staging times for boxes still in ROI at end of interval,
    # if they started within this interval
    current_frame = int(interval_end_time * fps)
    for box_id, start_frame in start_frames.items():
        start_time_s = start_frame / fps
        if interval_start_time <= start_time_s <= interval_end_time and box_id not in end_frames:
            staging_time_s = (current_frame - start_frame) / fps
            staging_times.append(staging_time_s)

    avg_staging_time = sum(staging_times) / len(staging_times) if staging_times else 0

    # Inflow rate (boxes per minute)
    interval_duration_minutes = (interval_end_time - interval_start_time) / 60
    inflow_rate = interval_inflow / interval_duration_minutes if interval_duration_minutes > 0 else 0

    # Outflow rate (boxes per minute)
    outflow_rate = interval_outflow / interval_duration_minutes if interval_duration_minutes > 0 else 0

    metrics = {
        'avg_area_utilization': round(avg_area_utilization, 2),
        'avg_utilized_area': round(avg_utilized_area, 2),
        'avg_staging_time': round(avg_staging_time, 2),
        'inflow_rate': round(inflow_rate, 2),
        'outflow_rate': round(outflow_rate, 2),
        'total_inflow': interval_inflow,
        'total_outflow': interval_outflow,
        'ROI_TOTAL_AREA': config.ROI_TOTAL_AREA # Include overall ROI area for context
    }
    metrics.update({f'avg_lane_{i+1}_utilization_percent': round(avg_lane_utilizations[f'lane_{i+1}_avg_util'], 2) for i in range(config.NUM_LANES)})
    return metrics

def generate_staging_excel(start_frames, end_frames, misplaced_boxes, current_time, fps, box_dimensions,
                           box_entry_direction, boxes_already_in_staging, box_last_seen_frame, box_exit_direction):
    """Generate staging times Excel file including misplaced boxes and boxes already in staging"""
    # Set the video recording start time (edit as needed if not starting from current time)
    video_start_time = datetime.now() - timedelta(seconds=current_time)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    data_rows = []

    # Add properly staged boxes (in ROI)
    for box_id, start_frame in start_frames.items():
        start_time = video_start_time + timedelta(seconds=start_frame / fps)
        start_time_str = start_time.strftime("%d:%m:%Y %H:%M:%S")
        if box_id in end_frames:
            end_frame = end_frames[box_id]
            end_time = video_start_time + timedelta(seconds=end_frame / fps)
            end_time_str = end_time.strftime("%d:%m:%Y %H:%M:%S")
            total_time = round((end_frame - start_frame) / fps, 2)
            status = "Exited ROI"
        else:
            end_time_str = 'Still in staging'
            current_frame = int(current_time * fps)
            if current_frame > start_frame:
                total_time = round((current_frame - start_frame) / fps, 2)
            else:
                total_time = 0
            status = "Still in staging"

        box_width, box_height = box_dimensions.get(box_id, (0, 0))
        box_area = box_width * box_height
        entry_direction = box_entry_direction.get(box_id, 'unknown')
        exit_direction = box_exit_direction.get(box_id, 'N/A')

        data_rows.append({
            'Box ID': box_id,
            'In Time': start_time_str,
            'Out Time': end_time_str,
            'Total Staging Time (s)': total_time,
            'Box Width (px)': box_width,
            'Box Height (px)': box_height,
            'Box Area (px²)': box_area,
            'Entry Direction': entry_direction,
            'Exit Direction': exit_direction,
            'Status': status
        })

    # Add misplaced boxes (outside ROI)
    for box_id, first_detected_frame in misplaced_boxes.items():
        if box_id in start_frames: # Skip if this box was later properly staged
            continue

        first_detected_time = video_start_time + timedelta(seconds=first_detected_frame / fps)
        first_detected_str = first_detected_time.strftime("%d:%m:%Y %H:%M:%S")

        # Calculate misplaced duration
        last_frame = box_last_seen_frame.get(box_id, first_detected_frame)
        last_seen_time = video_start_time + timedelta(seconds=last_frame / fps)
        last_seen_str = last_seen_time.strftime("%d:%m:%Y %H:%M:%S")
        total_misplaced_time = round((last_frame - first_detected_frame) / fps, 2)

        box_width, box_height = box_dimensions.get(box_id, (0, 0))
        box_area = box_width * box_height
        entry_direction = box_entry_direction.get(box_id, 'unknown')

        data_rows.append({
            'Box ID': box_id,
            'In Time': first_detected_str,
            'Out Time': last_seen_str,
            'Total Staging Time (s)': total_misplaced_time,
            'Box Width (px)': box_width,
            'Box Height (px)': box_height,
            'Box Area (px²)': box_area,
            'Entry Direction': entry_direction,
            'Exit Direction': 'N/A',
            'Status': 'Misplaced (Outside Staging Area)'
        })

    if data_rows:
        df_staging = pd.DataFrame(data_rows)
        staging_filename = f"box_staging_times_area_{current_timestamp}.xlsx"
        df_staging.to_excel(staging_filename, index=False)
        print(f"✅ Staging times Excel file '{staging_filename}' generated at {current_time:.1f}s")

        # Print a summary for verification
        properly_staged = [row for row in data_rows if 'Misplaced' not in row['Status']]
        misplaced = [row for row in data_rows if 'Misplaced' in row['Status']]
        already_in_staging = [row for row in properly_staged if row['Entry Direction'] == 'already_in_staging']

        if properly_staged:
            print("📋 Properly staged boxes:")
            for box in properly_staged:
                print(f"   Box {box['Box ID']}: {box['Total Staging Time (s)']}s staging time, Area: {box['Box Area (px²)']}px², Entry: {box['Entry Direction']}, Exit: {box['Exit Direction']}, Status: {box['Status']}")

        if already_in_staging:
            print("🏠 Boxes already in staging area (initially detected in ROI):")
            for box in already_in_staging:
                print(f"   Box {box['Box ID']}: {box['Total Staging Time (s)']}s staging time, Area: {box['Box Area (px²)']}px²")

        if misplaced:
            print("⚠️  Misplaced boxes (outside staging area):")
            for box in misplaced:
                print(f"   Box {box['Box ID']}: Misplaced for {box['Total Staging Time (s)']}s, Area: {box['Box Area (px²)']}px², Entry: {box['Entry Direction']}")


def generate_capacity_excel(interval_data, current_time):
    """Generate capacity utilization Excel file for 15-minute intervals, including lane-wise utilization."""
    video_start_time = datetime.now() - timedelta(seconds=current_time)
    current_timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    if interval_data:
        capacity_rows = []
        for i, interval in enumerate(interval_data):
            interval_start_timestamp = video_start_time + timedelta(seconds=interval['start_time'])
            interval_end_timestamp = video_start_time + timedelta(seconds=interval['end_time'])

            row = {
                'Interval': f"Interval {i+1} (Minutes {int(interval['start_time']/60)}-{int(interval['end_time']/60)})",
                'Start Time': interval_start_timestamp.strftime("%d:%m:%Y %H:%M:%S"),
                'End Time': interval_end_timestamp.strftime("%d:%m:%Y %H:%M:%S"),
                'Inflow Rate (boxes/min)': interval['inflow_rate'],
                'Outflow Rate (boxes/min)': interval['outflow_rate'],
                'Total Inflow': interval['total_inflow'],
                'Total Outflow': interval['total_outflow'],
                'Average Overall Area Utilization (%)': interval['avg_area_utilization'],
                'Average Overall Utilized Area (px²)': interval['avg_utilized_area'],
                'Average Staging Time (s)': interval['avg_staging_time'],
                'ROI Total Area (px²)': interval['ROI_TOTAL_AREA']
            }
            # Add lane-wise utilization to the row
            for lane_idx in range(config.NUM_LANES):
                row[f'Lane {lane_idx+1} Avg Utilization (%)'] = interval.get(f'avg_lane_{lane_idx+1}_utilization_percent', 0)

            capacity_rows.append(row)

        df_capacity = pd.DataFrame(capacity_rows)
        capacity_filename = f"roi_area_utilization_{current_timestamp}.xlsx"
        df_capacity.to_excel(capacity_filename, index=False)
        print(f"✅ Area utilization Excel file '{capacity_filename}' generated with {len(capacity_rows)} intervals")

def generate_excel_files(start_frames, end_frames, misplaced_boxes, interval_data, current_time, fps,
                         box_dimensions, box_entry_direction, boxes_already_in_staging,
                         lane_boundaries, lane_total_areas, box_last_seen_frame, box_exit_direction):
    """Generate both Excel files - staging times (with misplaced) and capacity utilization"""
    generate_staging_excel(start_frames, end_frames, misplaced_boxes, current_time, fps,
                           box_dimensions, box_entry_direction, boxes_already_in_staging,
                           box_last_seen_frame, box_exit_direction)
    generate_capacity_excel(interval_data, current_time)