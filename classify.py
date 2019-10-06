#Classify images (Copied from PyImageSearch)
#Kieran Hobden
#05-Oct-'19

#This document is not original work but a pre-made script with minor changes made for education purposes
#python classify.py --model pokedex.model --labelbin lb.pickle --image examples/charmander_counter.png

from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import pickle
import cv2
import os

#Construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
	help="path to label binarizer")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

#Load the image
image = cv2.imread(args["image"])
output = image.copy()
 
#Pre-process the image for classification (as before in train.py)
image = cv2.resize(image, (96, 96))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

#Load the trained CNN and the label binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
lb = pickle.loads(open(args["labelbin"], "rb").read())

#Classify the input image
print("[INFO] classifying image...")
proba = model.predict(image)[0]  #Array of probabilities with associated classes
idx = np.argmax(proba)  #Index of class with highest probability
label = lb.classes_[idx]   #Label of such index

#If input image filename contains the label text, mark prediction as correct
filename = args["image"][args["image"].rfind(os.path.sep) + 1:]
correct = "correct" if filename.rfind(label) != -1 else "incorrect"

#Build the label and draw the label on the image
label = "{}: {:.2f}% ({})".format(label, proba[idx] * 100, correct)
output = imutils.resize(output, width=400)  #Resize ouput image from 96x96 to 400x400 to view
cv2.putText(output, label, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)  #Apply text to image

#Show the output image
print("[INFO] {}".format(label))
cv2.imshow("Output", output)
cv2.waitKey(0)
