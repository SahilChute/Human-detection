import cv2
import numpy as np

# Define a callback function for mouse events
def select_roi(event, x, y, flags, params):
    global roi_corners, roi_selected
    if event == cv2.EVENT_LBUTTONDOWN and len(roi_corners) < 4:
        roi_corners.append((x, y))
    elif event == cv2.EVENT_RBUTTONDOWN:
        roi_corners = []
    elif event == cv2.EVENT_LBUTTONDBLCLK and len(roi_corners) == 4:
        roi_selected = True

# Create a named window for displaying the video
cv2.namedWindow('frame')
cv2.setMouseCallback('frame', select_roi)

# Capture frames from the video device until the user selects an ROI
cap = cv2.VideoCapture('C:\python_files\yolov7_dump\yolov7\input1.mp4')
roi_corners = []
roi_selected = False
while not roi_selected:
    ret, frame = cap.read()
    frame_copy = frame.copy()
    if len(roi_corners) == 4:
        roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        roi_pts = np.array([roi_corners], dtype=np.int32)
        cv2.fillPoly(roi_mask, roi_pts, (255,))
        frame_copy = cv2.bitwise_and(frame_copy, frame_copy, mask=roi_mask)
        cv2.polylines(frame_copy, [roi_pts], True, (0, 255, 0), 2)
        for i, corner in enumerate(roi_corners):
            cv2.circle(frame_copy, corner, 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, str(i+1), (corner[0]-10, corner[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    elif len(roi_corners) > 0:
        cv2.polylines(frame_copy, [np.array(roi_corners)], False, (0, 255, 0), 2)
        for i, corner in enumerate(roi_corners):
            cv2.circle(frame_copy, corner, 5, (0, 0, 255), -1)
            cv2.putText(frame_copy, str(i+1), (corner[0]-10, corner[1]-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
    cv2.imshow('frame', frame_copy)
    cv2.waitKey(1)

# Extract the ROI from the frame
roi_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
roi_pts = np.array([roi_corners], dtype=np.int32)
cv2.fillPoly(roi_mask, roi_pts, (255,))
roi = cv2.bitwise_and(frame, frame, mask=roi_mask)

# Display the ROI
cv2.imshow('ROI', roi)
cv2.waitKey(0)

# Release the video capture device and close all windows
cap.release()
cv2.destroyAllWindows()