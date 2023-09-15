import cv2
from random import randrange

trained_face_data = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#img = cv2.imread('zendaya.jpeg')
#img = cv2.imread('marvel_cast.jpg')
# to capture video from webcam
#webcam = cv2.VideoCapture(0) # 0 is for the default cam in your device.
webcam = cv2.VideoCapture('young_sheldon.mp4')

# Iterate forever over frames
while True:
    # Read the current frame
    successful_frame_read, frame = webcam.read() #successul_frame_read -> returns true or false, true if frame read successfully
    
    #cvtColor -> convert colour
    #RGB - Red, green, blue | In opencv, RGB is reverse - BGR
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #MultiScale - detects faces of different sizes in the input image
    # and the detected faces are returned as a list of rectangles. - gives coordinates of rectangles
    face_coordinates = trained_face_data.detectMultiScale(grayscaled_img)

    #(0, 255, 0) - BGR -> Green 
    #(255, 0, 0) - BGR -> Blue
    # 2 -> thickness of the rectangle  
    # this addition part below is like height and width added with the first coordinate
    #cv2.rectangle(img, (64, 75), (64+87, 75+87), (0, 255, 0), 2)
    # the above one is the manual one, but instead use the below
    #(x, y, w, h) = face_coordinates[1] # face coordinates are the list of lists
    for (x, y, w, h) in face_coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(256), randrange(256), randrange(256)), 10)
        # randrange is for generating different colours of boxes


    #cv2.namedWindow('Face Detector', cv2.WINDOW_NORMAL) #as for few pictures, it is getting zoomed in
    cv2.imshow('Face Detector', frame)
    key = cv2.waitKey(1) # waits for 1 millisecond and goes to other frame

    #Stop if Q key is pressed
    #ASCII of Q is 81 and ASCII of q is 113
    if key==81 or key==113:
        break

# Release the VideoCapture object
webcam.release()

print("Code completed")
