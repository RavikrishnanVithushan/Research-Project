import numpy as np
import cv2
from tensorflow.keras.models import load_model

model = load_model("backend/asl_model.h5")

alphabet = list("ABCDEFGHIKLMNOPQRSTUVWXY")

def predict_sign(image):

    image = cv2.resize(image,(28,28))
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    image = image.reshape(1,28,28,1)/255.0

    prediction = model.predict(image)

    class_index = np.argmax(prediction)

    return alphabet[class_index]