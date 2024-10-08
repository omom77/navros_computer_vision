from ultralytics import YOLO
import cv2
import matplotlib.pyplot as plt

import os
print("Current Working Directory:", os.getcwd())


# Load the YOLOv8 model with custom weights
model = YOLO('/home/om/Documents/navros_computer_vision/weights/best.pt')  # Replace with the path to your trained weights

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
while True:
    ret, img= cap.read()
    cv2.imshow('Webcam', img)

    if cv2.waitKey(1) == ord('q'):
        break

# Load an image to test the model
image = cv2.imread('/home/om/Documents/navros_computer_vision/navros_test_images/cone_test_1.jpg')  # Replace with the path to your image

# Perform inference on the image
results = model(image)

# Print the results
print(results)

# Draw the bounding boxes and labels on the image
annotated_image = results[0].plot()

# Display the image with detections
plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
plt.show()

# Optional: Save the image with detections
cv2.imwrite('/home/om/Documents/navros_computer_vision/results_test/annotated_image.jpg', annotated_image)
