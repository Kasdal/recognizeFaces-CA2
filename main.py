import os  # for interacting with the file system
import cv2  # for reading images
from flask_socketio import SocketIO  # for creating web sockets
import face_recognition  # for detecting and identifying faces in images
from flask import Flask, render_template, Response  # for creating a web application
import numpy as np  # for mathematical operations on arrays
import socket  # for getting the hostname and server address of the device running the script
import time
import threading
import functools
import waitress
# Determine the hostname and server address of the device running the script
hostname = socket.gethostname()
serverAddress = socket.gethostbyname(hostname)

# Create a Flask web application named app
recogApp = Flask(__name__)

# Create a SocketIO instance for the web application
socketioApp = SocketIO(recogApp)

# Set the path to the directory containing images
imgRepo = 'imgSrc'

# Create empty lists for images and image names
images = []
image_names = []

# Get a list of the files in the images directory
image_list = os.listdir(imgRepo)

# Print the server address to the console
print("http://" + serverAddress + ":8080")

# Loop through the files in the image list
for image in image_list:
    # Read the current image
    current_image = cv2.imread(f'{imgRepo}/{image}')

    # Add the image to the images list
    images.append(current_image)

    # Add the image name (without the extension) to the image_names list, in upper case
    image_names.append(os.path.splitext(image)[0].upper())


# Define a function to find the encodings of faces in a list of images
def find_encodings(images_list):
    # Create an empty list for the encodings
    encoding_list = []

    # Loop through the images
    for img in images_list:
        # Convert the image to RGB color space
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Calculate the face encoding for the image
        encoding = face_recognition.face_encodings(img)[0]

        # Add the encoding to the list
        encoding_list.append(encoding)

    # Return the list of encodings
    return encoding_list


# Find the encodings for the images in the images list
encode_list_known = find_encodings(images)

# Create a VideoCapture object to capture video frames from the webcam
capture = cv2.VideoCapture(0)

def measure_distance(object_points, image_points, image_size):
    # Calculate camera matrix and distortion coefficients
    _, camera_matrix, distortion_coeffs, _, _ = cv2.calibrateCamera(object_points, image_points, image_size, None, None)

    # Project object points onto the image plane
    _, _, _, image_points = cv2.projectPoints(object_points, camera_matrix, distortion_coeffs, None, None)

    # Calculate distance from camera using formula: distance = (focal length * object height) / (object point y - image point y)
    focal_length = camera_matrix[0][0]
    object_height = 1  # Set object height to 1 for simplicity
    object_point_y = object_points[0][0][1]
    image_point_y = image_points[0][0][1]
    distance = (focal_length * object_height) / (object_point_y - image_point_y)

    return distance


@functools.lru_cache(maxsize=None)
def recognise_faces():
    # Define the font to be used for the text
    font = cv2.FONT_HERSHEY_SIMPLEX

    # Initialize frame count and start time
    frame_count = 0
    start_time = time.time()

    while True:
        # Capture a frame from the webcam
        success, img = capture.read()
        # Check if the frame was successfully captured
        if img is not None:
            # Increment frame count
            frame_count += 1

            # Resize the frame to a quarter of its original size and convert it to RGB color space
            image_small = cv2.resize(img, (0, 0), None, 0.25, 0.25)
            image_small = cv2.cvtColor(image_small, cv2.COLOR_BGR2RGB)

            # Define a thread for face detection and recognition
            def detect_and_recognize_faces():
                # Detect the locations of faces in the image
                face_locations = face_recognition.face_locations(image_small)

                # Calculate the face encodings for the detected faces
                face_encodings = face_recognition.face_encodings(image_small, face_locations)

                # Loop through the detected faces
                for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                    # Check if the face is recognized
                    if len(face_locations) > 0:
                        # Check if the face encoding is in the list of known face encodings
                        matches = face_recognition.compare_faces(encode_list_known, face_encoding)
                        name = "UNKNOWN"
                        # If a match is found, get the index of the match in the list
                        if True in matches:
                            matched_idx = matches.index(True)
                            # Get the name of the person with the matching face encoding
                            name = image_names[matched_idx]
                    else:
                        name = "UNKNOWN"

                    # Scale the face location coordinates back up to the original size of the image
                    top *= 4
                    right *= 4
                    bottom *= 4
                    left *= 4

                    # Draw a rectangle around the face
                    cv2.rectangle(img, (left, top), (right, bottom), (0, 0, 255), 2)
                    cv2.putText(img, name, (left + 6, bottom - 6), font, 1.0, (0, 0, 255), 1, cv2.LINE_AA)

            # Start the face detection and recognition thread
            thread = threading.Thread(target=detect_and_recognize_faces)
            thread.start()

            # Calculate FPS
            elapsed_time = time.time() - start_time
            fps = frame_count / elapsed_time


            # Draw FPS on the frame
            cv2.putText(img, f'FPS: {fps:.2f}', (10, 30), font, 1, (0, 0, 255), 1, cv2.LINE_AA)


            # Display the frame with FPS
            # cv2.imshow('Frame', img)

            # Reset frame count and start time if elapsed time is greater than 1 second
            if elapsed_time > 1:
                frame_count = 0
                start_time = time.time()

                #socketioApp.sleep(0.1)

            # Convert the image to JPEG format and send it to the client
            image_encoded = cv2.imencode('.jpg', img)[1]
            socketioApp.emit('image', image_encoded.tobytes())

            # Encode the frame in JPEG format
        ret, buffer = cv2.imencode('.jpg', img)

        # Convert the encoded frame to a bytes object
        img = buffer.tobytes()
        # Send the frame to the client
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + img + b'\r\n')

        if cv2.waitKey(1) == ord('q'):
            break
        #
    capture.release()  # turn off cam
    cv2.destroyAllWindows()  # close all windows


# Define a route for the index page
@recogApp.route('/video_stream')
def video_stream():
    return Response(recognise_faces(), mimetype='multipart/x-mixed-replace; boundary=frame')

@recogApp.route('/')
def index():
    return render_template('index.html')

# Run the web application
def run():
    socketioApp.run(recogApp)


if __name__ == '__main__':
    socketioApp.run(recogApp)
