"""
Object Detection Engine using YOLOv3 with Darknet & OpenCV DNN
Features:
- Robust multi-version OpenCV compatibility (handles scalar, tuple, ndarray NMS results)
- Modern graphical overlays with semi-transparent contrast badges
- On-screen Heads-Up Display (HUD) showing FPS, Object Count, Audio status, and Hotkeys
- Snapshot saving utility with timestamps
"""

import cv2 as cv
import numpy as np
import time
import os
from datetime import datetime


def listBoundingBoxes(cnn_outs_layer, image_height, image_width, threshold_conf=0.5):
    """Extract bounding boxes, confidences, and class IDs from YOLO output layers."""
    box_array = []
    confidence_array = []
    class_ids_array = []

    for out in cnn_outs_layer:
        for obj_detection in out:
            detection_scores = obj_detection[5:]
            class_id = int(np.argmax(detection_scores))
            confidence_value = float(detection_scores[class_id])
            
            if confidence_value > threshold_conf:
                # Scale coordinates back to original image dimensions
                bound_box = obj_detection[0:4] * np.array([image_width, image_height, image_width, image_height])
                center_x, center_y, box_width, box_height = bound_box.astype('int')

                # Calculate top-left corner
                xx = int(center_x - (box_width / 2))
                yy = int(center_y - (box_height / 2))

                box_array.append([xx, yy, int(box_width), int(box_height)])
                confidence_array.append(confidence_value)
                class_ids_array.append(class_id)

    return box_array, confidence_array, class_ids_array


def labelsBoundingBoxes(image, bounding_boxes, confidences, class_ids, ids, label_colors, class_labels):
    """Draw stylish bounding boxes and high-contrast labels on the image."""
    detected_classes = []
    
    if ids is None or len(ids) == 0:
        return image, detected_classes

    # Ensure ids is a 1D flat array regardless of OpenCV version format
    flat_ids = np.array(ids).flatten()

    for i in flat_ids:
        if i >= len(bounding_boxes) or i >= len(class_ids):
            continue

        xx, yy, width, height = bounding_boxes[i]
        cid = class_ids[i]
        conf = confidences[i]
        
        # Color for this class
        color = [int(c) for c in label_colors[cid % len(label_colors)]]
        
        # Draw main bounding box with rounded corner accents
        cv.rectangle(image, (xx, yy), (xx + width, yy + height), color, 2)
        
        # Corner accent markers
        corner_len = min(20, width // 4, height // 4)
        if corner_len > 4:
            cv.line(image, (xx, yy), (xx + corner_len, yy), color, 4)
            cv.line(image, (xx, yy), (xx, yy + corner_len), color, 4)
            cv.line(image, (xx + width, yy), (xx + width - corner_len, yy), color, 4)
            cv.line(image, (xx + width, yy), (xx + width, yy + corner_len), color, 4)
            cv.line(image, (xx, yy + height), (xx + corner_len, yy + height), color, 4)
            cv.line(image, (xx, yy + height), (xx, yy + height - corner_len), color, 4)
            cv.line(image, (xx + width, yy + height), (xx + width - corner_len, yy + height), color, 4)
            cv.line(image, (xx + width, yy + height), (xx + width, yy + height - corner_len), color, 4)

        label_name = class_labels[cid] if cid < len(class_labels) else f"Object {cid}"
        detected_classes.append(label_name)
        
        # Label text badge: Name and Confidence %
        label_text = f"{label_name.upper()} {int(conf * 100)}%"
        font = cv.FONT_HERSHEY_DUPLEX
        font_scale = 0.55
        thickness = 1
        
        (text_w, text_h), baseline = cv.getTextSize(label_text, font, font_scale, thickness)
        
        # Position label above bounding box, or inside if too close to top
        label_y = yy - 8 if yy - text_h - 10 > 0 else yy + text_h + 10
        badge_top_left = (xx, label_y - text_h - 6)
        badge_bottom_right = (xx + text_w + 10, label_y + 4)
        
        # Draw solid label background badge
        cv.rectangle(image, badge_top_left, badge_bottom_right, color, -1)
        
        # Draw text in white or dark for high contrast
        text_color = (0, 0, 0) if (color[0]*0.299 + color[1]*0.587 + color[2]*0.114) > 160 else (255, 255, 255)
        cv.putText(image, label_text, (xx + 5, label_y), font, font_scale, text_color, thickness, cv.LINE_AA)

    return image, detected_classes


def detectObject(CNNnet, total_layer_names, image_height, image_width, image, name_colors, class_labels,
                 Boundingboxes=None, confidence_value=None, class_ids=None, ids=None, detect=True,
                 conf_threshold=0.5, nms_threshold=0.3, input_size=(416, 416)):
    """
    Main detection pipeline:
    - Preprocesses image into Darknet blob
    - Runs YOLO inference
    - Performs Non-Maximum Suppression (NMS)
    - Draws visual overlays
    """
    cls = []
    if detect and CNNnet is not None:
        blob_object = cv.dnn.blobFromImage(image, 1 / 255.0, input_size, swapRB=True, crop=False)
        CNNnet.setInput(blob_object)
        cnn_outs_layer = CNNnet.forward(total_layer_names)
        
        Boundingboxes, confidence_value, class_ids = listBoundingBoxes(
            cnn_outs_layer, image_height, image_width, conf_threshold
        )
        
        if len(Boundingboxes) > 0:
            ids = cv.dnn.NMSBoxes(Boundingboxes, confidence_value, conf_threshold, nms_threshold)
        else:
            ids = ()
            
        image, cls = labelsBoundingBoxes(
            image, Boundingboxes, confidence_value, class_ids, ids, name_colors, class_labels
        )

    return image, cls, Boundingboxes, confidence_value, class_ids, ids


def drawHUD(image, fps=0.0, object_count=0, voice_info="Voice: Active", conf_thresh=0.5, is_paused=False):
    """
    Draw a sleek Heads-Up Display (HUD) banner on the frame:
    - Top bar: FPS, Status, Detection Count, Voice Info
    - Bottom bar: Keyboard shortcuts guide
    """
    h, w = image.shape[:2]
    
    # 1. Top HUD Bar (Translucent dark overlay)
    overlay = image.copy()
    top_bar_height = 42
    cv.rectangle(overlay, (0, 0), (w, top_bar_height), (20, 20, 24), -1)
    
    # 2. Bottom Shortcut Bar
    bottom_bar_height = 30
    cv.rectangle(overlay, (0, h - bottom_bar_height), (w, h), (20, 20, 24), -1)
    
    # Blend overlay with 75% opacity
    cv.addWeighted(overlay, 0.75, image, 0.25, 0, image)
    
    # Text fonts
    font = cv.FONT_HERSHEY_SIMPLEX
    
    # Status Dot & Title
    status_color = (0, 165, 255) if is_paused else (0, 255, 128)  # Orange if paused, Bright Green if live
    status_text = "PAUSED" if is_paused else "LIVE STREAM"
    cv.circle(image, (16, 21), 6, status_color, -1)
    cv.putText(image, status_text, (28, 26), font, 0.55, (255, 255, 255), 1, cv.LINE_AA)
    
    # FPS Counter
    fps_text = f"FPS: {fps:4.1f}"
    cv.putText(image, fps_text, (180, 26), font, 0.55, (100, 255, 255), 1, cv.LINE_AA)
    
    # Detected Objects Count
    count_text = f"Objects: {object_count}"
    cv.putText(image, count_text, (310, 26), font, 0.55, (255, 200, 100), 1, cv.LINE_AA)
    
    # Confidence Threshold
    conf_text = f"Conf: {int(conf_thresh * 100)}%"
    cv.putText(image, conf_text, (460, 26), font, 0.55, (220, 220, 220), 1, cv.LINE_AA)
    
    # Voice Info (Muted or Active)
    voice_color = (100, 100, 255) if "Muted" in voice_info else (120, 255, 120)
    cv.putText(image, voice_info, (max(w - 260, 580), 26), font, 0.5, voice_color, 1, cv.LINE_AA)
    
    # Bottom Help Bar
    shortcuts_text = "[Q] Quit  |  [M/V] Mute Voice  |  [E] Toggle Engine  |  [S] Save Snapshot  |  [SPACE] Pause"
    cv.putText(image, shortcuts_text, (15, h - 9), font, 0.42, (200, 200, 200), 1, cv.LINE_AA)
    
    return image


def saveSnapshot(image, output_dir='snapshots'):
    """Save the current annotated frame to disk with a timestamped filename."""
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(output_dir, f"detection_{timestamp}.jpg")
    cv.imwrite(filepath, image)
    return filepath


def displayImage(image, window_title="Detection Result"):
    """Display static image and wait for keypress."""
    cv.imshow(window_title, image)
    cv.waitKey(0)
    cv.destroyAllWindows()
