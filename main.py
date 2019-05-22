import numpy as np
import cv2
import model
import os
import skimage.transform
model.generate_caption_init()
count=0
print("Enter 1 to generate captioned images in folder\nEnter 2 to generate captions in real-time via webcam\n")
entry = input()

if entry == '1':
    model.generate_caption_bulk()



if entry == '2':
    cap = cv2.VideoCapture(cv2.CAP_DSHOW)
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        count += 1
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if count == 120:
            cv2.imwrite('Images/image.png', frame)
            model.generate_caption_live()
            count = 0

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
