## for nose detection download this:
# https://raw.githubusercontent.com/opencv/opencv_contrib/master/modules/face/data/cascades/haarcascade_mcs_nose.xml

## for mouth detection download this:
# https://raw.githubusercontent.com/peterbraden/node-opencv/master/data/haarcascade_mcs_mouth.xml


# Paths to Cascade Classifiers
face_casc_path = "/Users/csstnns/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_frontalface_default.xml"
eye_casc_path = "/Users/csstnns/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_eye.xml"
nose_casc_path = "/Users/csstnns/anaconda3/lib/python3.7/site-packages/cv2/data/haarcascade_mcs_nose.xml"

#import required libraries
from keras.models import load_model
import cv2
import sys
import numpy as np
#load the CNN model we saved
model = load_model('face_rec.h5')
#we know the label to name maps so let's use them
label_map = {0: 'darda', 1: 'labenm', 2: 'Noureddin', 3: 'sjcutt'}

# create a Cascade Classifier object for face, eye and nose
faceCascade = cv2.CascadeClassifier(face_casc_path)
eyeCascade = cv2.CascadeClassifier(eye_casc_path)
noseCascade = cv2.CascadeClassifier(nose_casc_path)

## Connect to WebCam and start video capture
video_capture = cv2.VideoCapture(0)

## Videos are made of frames so we loop through the recording
## one frame at a time
while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()
    # Write the frame into the file 'output.avi'
    #out.write(frame)
    
    # convert from color to grayscale .. notice BGR not RGB
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ## detect any faces in this frame
    ## it'll return starting (x,y) and width and height of face rectangle
    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    font = cv2.FONT_HERSHEY_DUPLEX
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        #grab the face window on the frame to feed it into the CNN model
        roi = frame[y:y+h, x:x+w]
        #notice w,h are swapped
        roi = cv2.resize(roi,(180, 200))
        #add 3d dimension so it's a tensor to feed into the model
        test_img = np.expand_dims(roi, axis=0)
        test_img = test_img/255 ## normalize image
        ## get the class probability and then the correct label
        y_prob = model.predict(test_img)
        max_prob = max(y_prob[0])
        max_prob_id = y_prob.argmax(axis=-1)[0]
        label = label_map[max_prob_id]

        #cv2.imwrite('roi.png', roi)#if you wish to save the current frame
        cv2.rectangle(frame, (x, y), (x+w, y+h), (100, 255, 255), 2)
        cv2.putText(frame, label+" : "+str(max_prob * 100)+" %", (x + 6, y - 6), font, 1.5, (255, 0, 0), 2)
    ## if you wish to detect eyes or noses
    eyes = eyeCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (255, 0, 0), 2)
       
    noses = noseCascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (ex, ey, ew, eh) in noses:
        cv2.rectangle(frame, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    # Display the resulting frame
    cv2.imshow('Video', frame)
    ## this is to wait for the key press 'q' and exist
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()
