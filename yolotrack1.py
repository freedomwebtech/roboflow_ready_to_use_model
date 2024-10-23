import cv2
import requests
import base64
from inference_sdk import InferenceHTTPClient

# Initialize the inference client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="LYIvJJkEVL0VfDTTwHAg"
)

# Function to encode image to base64
def encode_image(image):
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode('utf-8')

# Open a connection to the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

count = 0
while True:
    # Capture frame-by-frame from the webcam
    ret, frame = cap.read()
    frame_resized = cv2.resize(frame, (640, 480))
    count += 1
    if count % 3 != 0:
        continue
    if not ret:
        print("Failed to capture image")
        break

    # Encode the image as base64 for the API request
    img_base64 = encode_image(frame_resized)

    # Perform inference
    results = CLIENT.infer(img_base64, model_id="face-detect-vracc/1")
    print(results)
    
    # Extract predictions (bounding boxes) and class labels
    if 'predictions' in results:
        for pred in results['predictions']:
            x = pred['x']  # center x-coordinate
            y = pred['y']  # center y-coordinate
            width = pred['width']
            height = pred['height']
            
            # Extract class label if present
            class_label = pred.get('class')  # Get the class label ('real' or 'fake')

            # Convert to top-left (x1, y1) and bottom-right (x2, y2) for rectangle drawing
            x1 = int(x - width / 2)
            y1 = int(y - height / 2)
            x2 = int(x + width / 2)
            y2 = int(y + height / 2)

            # Check if the class is 'real' or 'fake' and apply the corresponding rectangle and label
            if class_label == 'real':
                # Draw a green rectangle for real faces
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green rectangle
                # Display the class label (real) above the rectangle
                cv2.putText(frame_resized, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            elif class_label == 'fake':
                # Draw a red rectangle for fake faces
                cv2.rectangle(frame_resized, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red rectangle
                # Display the class label (fake) above the rectangle
                cv2.putText(frame_resized, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
                
    # Display the resulting frame
    cv2.imshow('Webcam - Face Detection', frame_resized)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
