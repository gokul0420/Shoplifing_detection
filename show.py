import time

import cv2
import numpy as np

print("hello")
# Initialize video capture
video1 = ['summa.mp4','twoThirttu.mp4','bagThiruttu.mp4','thiruttu1.mp4','summawalk.mp4','summaWalking.mp4','thiruttu.mp4', 'thiruttukundi.mp4']
for video_path in video1:
    cap = cv2.VideoCapture(f"Samples\inputs\inout1.mp4")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Define codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    output_video = cv2.VideoWriter('output_video1.avi', fourcc, fps, (frame_width, frame_height))

    # Parameters for motion detection
    min_contour_area = 1000
    motion_frames = []

    # Parameters for shoplifting detection
    shoplifting_threshold = 0.02  # Adjust this threshold as needed
    shoplifting_frames = []
    shoplifting_delay = 2  # Adjust this delay (in seconds)

    shoplifting_start_time = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale for motion detection
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.GaussianBlur(gray_frame, (21, 21), 0)

        motion_frames.append(gray_frame)

        if len(motion_frames) > 2:
            frame_delta = cv2.absdiff(motion_frames[-2], motion_frames[-1])
            thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)[1]
            thresh = cv2.dilate(thresh, None, iterations=2)

            contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            detected_changes = []

            for contour in contours:
                if cv2.contourArea(contour) < min_contour_area:
                    continue
                x, y, w, h = cv2.boundingRect(contour)
                detected_changes.append((x, y, w, h))

            # Process each detected change
            for detection in detected_changes:
                x, y, w, h = detection  # Extract bounding box coordinates

                # Draw bounding box on frame
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

                # Check for potential shoplifting
                roi = gray_frame[y:y + h, x:x + w]
                _, shoplifting_thresh = cv2.threshold(roi, 25, 255, cv2.THRESH_BINARY)
                shoplifting_frames.append(shoplifting_thresh)

                if len(shoplifting_frames) > 30:
                    avg_frame_diff = np.zeros_like(shoplifting_frames[-1], dtype=np.float64)  # Use float64 for division
                    for f in shoplifting_frames[-30:]:
                        f_resized = cv2.resize(f, shoplifting_frames[-1].shape[::-1])
                        avg_frame_diff += f_resized
                    avg_frame_diff /= 30.0
                    avg_frame_diff = (avg_frame_diff * 255).astype(np.uint8)  # Convert back to uint8

                    if shoplifting_start_time is None:
                        shoplifting_start_time = time.time()
                    if np.max(
                            avg_frame_diff) > shoplifting_threshold and time.time() - shoplifting_start_time > shoplifting_delay:
                        print(np.max(
                            avg_frame_diff), 'siebg')
                        # Potential shoplifting detected
                        cv2.putText(frame, "Shoplifting Detected", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Show frame with bounding boxes
        cv2.imshow('Frame', frame)

        # Write frame to output video
        output_video.write(frame)

        # Break the loop if 'q' key is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release video capture and writer
    cap.release()
    output_video.release()

    cv2.destroyAllWindows()