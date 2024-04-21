import cv2
import numpy as np  

def draw_triangle_on_face(image_path):
    # Load the image
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the pre-trained face detection model
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        # Points for the triangle, you can adjust these according to your requirements
        point1 = (x + w // 2, y)
        point2 = (x, y + h)
        point3 = (x + w, y + h)

        # Draw the triangle
        triangle_cnt = np.array([point1, point2, point3])
        cv2.drawContours(img, [triangle_cnt], 0, (0, 255, 0), -1)

    # Display the image
    cv2.imshow('Image with Triangle Mask', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Usage
draw_triangle_on_face('0.jpg')
