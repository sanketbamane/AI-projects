import cv2
import numpy as np
import os

# create a synthetic image with 'defects'
img = np.full((400,400,3), 255, dtype=np.uint8)
cv2.circle(img, (100,100), 20, (0,0,255), -1)  # defect 1
cv2.rectangle(img, (250,250), (280,280), (0,0,255), -1)  # defect 2
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
_,th = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x,y,w,h = cv2.boundingRect(c)
    cv2.rectangle(img, (x,y),(x+w,y+h),(0,255,0),2)
out = 'src/defect_demo.png'
cv2.imwrite(out, img)
print('Saved demo image to', out)
