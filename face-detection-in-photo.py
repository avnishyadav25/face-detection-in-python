import cv2
import sys

#Get user supplied values
imagePath = "trial.jpg" 
cascPath = "haarcascade_frontalface_default.xml"

#Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)

#read the image and convert it int gray scale.
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Detect faces
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor = 1.3,
    minNeighbors = 5,
    minSize = (30,30),
    flags = cv2.CASCADE_SCALE_IMAGE
)

print("Found {0} faces!".format(len(faces)))

#Draw a rectangle around the faces
for(x,y,w,h) in faces:
    cv2.rectangle(image,(x,y),(x+w, y+h),(0,255,0),2)

cv2.imshow("Faces Found", image)
cv2.waitKey(0)

