#   STEP 2 TRAIN MODEL FROM THE CREATED TRAINING DATA
from sys import path
import cv2
import numpy as np
from os import listdir
from os.path  import isfile, join
# Get training data we previously made
def trainModel(path):
    
    data_path = path
    onlyfiles = [f for f in listdir(data_path) if isfile(join(data_path, f))]
    # Create arrays from training data and labels
    Training_Data, Labels = [], []

    # Open training images in our datapath
    # Create a numpy array for training data
    for i, files in enumerate(onlyfiles):
        image_path = data_path + onlyfiles[i]
        images = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        Training_Data.append(np.asarray(images, dtype=np.uint8))
        Labels.append(i)
    
    # Create a numpy array for both training data and labels
    Labels = np.asarray(Labels, dtype=np.int32)
    
    model = cv2.face_LBPHFaceRecognizer.create()
    # Let's train our model 
    model.train(np.asarray(Training_Data), np.asarray(Labels))
    return model
  

face1_model = trainModel("./faces/face1/")
print("Face1_model trained sucessefully")
face2_model = trainModel("./faces/face2/")
print("Face2_model trained sucessefully")