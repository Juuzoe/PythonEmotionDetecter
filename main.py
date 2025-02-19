"""
Python Emotion Detector

Press 'q' to quit the application.
"""

import cv2
from deepface import DeepFace


def main():
    # Open the default webcam (device index 0)
    # Change if you have multiple cameras.
    video_capture = cv2.VideoCapture(0)
    if not video_capture.isOpened():
        print("Error: Unable to open the camera.")
        return

    print("Starting real-time emotion detection. Press 'q' to quit.")

    frame_counter = 0
    # Analyses every 5th frame for performance reasons. Change if you want a higher frequency.
    frames_between_analysis = 5
    latest_analysis = None

    while True:
        ret, frame = video_capture.read()
        if not ret:
            print("Error: Unable to capture a frame from the camera. Exiting...")
            break

        frame_counter += 1

      
        if frame_counter % frames_between_analysis == 0:
            try:
                analysis_result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
                if isinstance(analysis_result, list) and len(analysis_result) > 0:
                    latest_analysis = analysis_result[0]
                else:
                    latest_analysis = analysis_result
            except Exception as error:
                print("Error during emotion analysis:", error)
                latest_analysis = None

        if latest_analysis is not None:
            dominant_emotion = latest_analysis.get("dominant_emotion", "N/A")
            face_region = latest_analysis.get("region")

            if face_region is not None:
                x = int(face_region.get("x", 0))
                y = int(face_region.get("y", 0))
                width = int(face_region.get("w", 0))
                height = int(face_region.get("h", 0))
                # Draws a green rectangle around the detected face. I have tested both on laptop and desktop but you can modify if it doesn't draw properly.
                cv2.rectangle(frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                cv2.putText(frame, dominant_emotion, (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            else:
                # If face area is not returned, it displays the emotion in the top-left corner. Also modify if needed.
                cv2.putText(frame, dominant_emotion, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        cv2.imshow("Real-Time Emotion Detector", frame)

        # Exits the loop when 'q' is pressed. Change key or remove this if used in a project.
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    video_capture.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
