import cv2
import numpy as np

def calculate_box_area_in_roi(box_coords, roi_bounds):
    """
    Calculate the area of a box that overlaps with a RECTANGULAR ROI.
    Returns the overlapping area in pixels²
    """
    x1, y1, x2, y2 = box_coords
    roi_x1, roi_y1, roi_x2, roi_y2 = roi_bounds

    # Calculate intersection coordinates
    intersect_x1 = max(x1, roi_x1)
    intersect_y1 = max(y1, roi_y1)
    intersect_x2 = min(x2, roi_x2)
    intersect_y2 = min(y2, roi_y2)

    # Check if there's an intersection
    if intersect_x1 < intersect_x2 and intersect_y1 < intersect_y2:
        intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
        return intersect_area
    else:
        return 0

def calculate_area_utilization(boxes_data, target_area_bounds, target_total_area):
    """
    Calculate area utilization from given boxes within specified RECTANGULAR target bounds.
    This function is now primarily for the overall ROI. Lane utilization is calculated differently in main.py.
    boxes_data: list of (track_id, box_coords) tuples
    target_area_bounds: (x1, y1, x2, y2) of the target area (e.g., ROI)
    target_total_area: Total area of the target region in pixels²
    """
    total_utilized_area = 0

    for track_id, box_coords in boxes_data:
        box_area_in_target = calculate_box_area_in_roi(box_coords, target_area_bounds)
        total_utilized_area += box_area_in_target

    # Calculate utilization percentage
    utilization_percentage = (total_utilized_area / target_total_area) * 100 if target_total_area > 0 else 0
    return utilization_percentage, total_utilized_area

def determine_entry_direction(track_id, center_x, is_in_roi, box_first_position, box_entry_direction, boxes_already_in_staging, frame_count, width, fps):
    """
    Determine if box entered from left, right, or was already in staging area
    """
    if track_id not in box_first_position:
        box_first_position[track_id] = center_x

        # If box is detected in ROI on first detection, mark as already in staging
        if is_in_roi:
            box_entry_direction[track_id] = 'already_in_staging'
            boxes_already_in_staging.add(track_id)
            print(f"📋 Box ID {track_id} detected ALREADY IN STAGING AREA at frame {frame_count} ({frame_count/fps:.1f}s)")
        else:
            # Determine entry direction based on which side of frame the box first appeared
            frame_center = width // 2
            if center_x < frame_center:
                box_entry_direction[track_id] = 'left'
            else:
                box_entry_direction[track_id] = 'right'

    return box_entry_direction.get(track_id, 'unknown')

def get_lane_for_box(box_center, lane_polygons):
    """
    Determines which lane polygon a box's center falls into.
    Returns the lane index (0-based) or -1 if outside all lanes.
    """
    for i, polygon in enumerate(lane_polygons):
        # Use pointPolygonTest to check if the center point is inside the polygon
        if cv2.pointPolygonTest(polygon, box_center, False) >= 0:
            return i
    return -1

def determine_exit_direction(center_x, width):
    """
    Determine if a box exited from the left or right side of the frame.
    """
    frame_center = width // 2
    if center_x < frame_center:
        return 'left'
    else:
        return 'right'