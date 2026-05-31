# Emotion Detector — real-time facial emotion recognition using DeepFace and OpenCV
import cv2
import requests
import datetime
import matplotlib.pyplot as plt
import time
import json
import logging

# ---------------------------
# Logging Setup
# ---------------------------
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),  # Logs to the console
        # To log to a file, uncomment the next line:
        # logging.FileHandler("emotion_detector.log")
    ]
)

# ---------------------------
# Linked List Data Structure
# ---------------------------
class EmotionNode:
    def __init__(self, timestamp, primary_emotion, score, camera_name):
        self.timestamp = timestamp
        self.primary_emotion = primary_emotion
        self.score = score
        self.camera_name = camera_name
        self.next = None

class EmotionLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
    
    def append(self, timestamp, primary_emotion, score, camera_name):
        new_node = EmotionNode(timestamp, primary_emotion, score, camera_name)
        if not self.head:
            self.head = new_node
            self.tail = new_node
        else:
            self.tail.next = new_node
            self.tail = new_node

    def to_list(self):
        """Convert linked list data to a list for easy processing (e.g., for plotting)."""
        result = []
        current = self.head
        while current:
            result.append((current.timestamp, current.primary_emotion, current.score, current.camera_name))
            current = current.next
        return result

# ---------------------------
# Microsoft Face API Setup
# ---------------------------
def get_emotions(face_img):
    """
    Sends the face image to the Microsoft Face API and returns the emotion scores.
    :param face_img: The cropped face image (as a NumPy array)
    :return: A dictionary of emotion scores or None if an error occurs.
    """
    # Convert the face image to JPEG bytes
    ret, buf = cv2.imencode('.jpg', face_img)
    if not ret:
        logging.error("Failed to encode image to JPEG.")
        return None
    img_bytes = buf.tobytes()
    
    # --- Replace with your Face API details ---
    subscription_key = "YOUR_MICROSOFT_FACE_API_KEY"  # <-- Insert your API key
    face_api_url = "https://<your-region>.api.cognitive.microsoft.com/face/v1.0/detect"  # <-- Replace <your-region>
    
    headers = {
        'Ocp-Apim-Subscription-Key': subscription_key,
        'Content-Type': 'application/octet-stream'
    }
    params = {
        'returnFaceAttributes': 'emotion'
    }
    
    response = requests.post(face_api_url, params=params, headers=headers, data=img_bytes)
    
    if response.status_code != 200:
        logging.error("Error calling Face API: %s", response.json())
        return None
    
    data = response.json()
    if data and "faceAttributes" in data[0]:
        emotions = data[0]["faceAttributes"]["emotion"]
        return emotions
    else:
        logging.warning("No emotion data found in API response.")
        return None

# ---------------------------
# Plotting Emotion Trends
# ---------------------------
def plot_emotion_trends(emotion_data):
    """
    Plots a bar chart of the frequency of primary emotions detected.
    :param emotion_data: List of tuples in the format (timestamp, primary_emotion, score, camera_name)
    """
    # Count frequency of each primary emotion
    emotion_counts = {}
    for entry in emotion_data:
        primary_emotion = entry[1]
        emotion_counts[primary_emotion] = emotion_counts.get(primary_emotion, 0) + 1
    
    # Prepare data for plotting
    emotions = list(emotion_counts.keys())
    counts = list(emotion_counts.values())
    
    plt.figure(figsize=(10,6))
    plt.bar(emotions, counts, color='skyblue')
    plt.xlabel("Emotions")
    plt.ylabel("Frequency")
    plt.title("Emotion Trends Over Time")
    plt.show()

# ---------------------------
# Main Application Loop
# ---------------------------
def main():
    # Use default webcam (camera index 0)
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        logging.error("Cannot open the default camera")
        return
    
    # Define camera name (modify if needed)
    camera_name = "Default Webcam"
    
    # Load OpenCV's pre-trained Haar Cascade for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    
    # Initialize the linked list for logging emotion data
    emotion_log = EmotionLinkedList()
    
    logging.info("Starting emotion detection. Press 'q' to quit.")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logging.error("Failed to capture frame from camera. Exiting...")
            break
        
        # Convert frame to grayscale for face detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        for (x, y, w, h) in faces:
            # Draw a rectangle around the detected face
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # Crop the face region from the frame
            face_img = frame[y:y+h, x:x+w]
            
            # Get emotion scores for the face via the API
            emotions = get_emotions(face_img)
            
            if emotions:
                # Determine the primary emotion (the one with the highest score)
                primary_emotion = max(emotions, key=emotions.get)
                score = emotions.get(primary_emotion)
                # Overlay the primary emotion on the video frame
                cv2.putText(frame, primary_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 
                            0.9, (36, 255, 12), 2)
                
                # Timestamp for logging
                timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                # Append data to the linked list
                emotion_log.append(timestamp, primary_emotion, score, camera_name)
                
                # Log to the console
                logging.info("Camera: %s | Time: %s | Emotion: %s (Score: %.2f)",
                             camera_name, timestamp, primary_emotion, score)
        
        # Display the video with overlaid emotion information
        cv2.imshow("AI-Based Emotion Detector", frame)
        
        # Break out of the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        # A short delay to manage API call frequency and performance
        time.sleep(0.1)
    
    # Release the webcam and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()
    
    # Convert the linked list data to a list for further analysis/plotting
    emotion_data = emotion_log.to_list()
    logging.info("Total emotion entries logged: %d", len(emotion_data))
    
    # Plot the collected emotion trends
    plot_emotion_trends(emotion_data)

if __name__ == "__main__":
    main()

# test1
