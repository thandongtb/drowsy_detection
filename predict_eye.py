from keras.preprocessing import image as keras_image
from keras.models import model_from_json
from imutils import face_utils
import numpy as np
import cv2
import imutils
import dlib
import argparse

# Command to run: python predict_eye.py -img temp/closed_eye_0013.BMP_face_1.jpg

def showImage(img, weight, height):
    screen_res = weight, height
    scale_width = screen_res[0] / img.shape[1]
    scale_height = screen_res[1] / img.shape[0]
    scale = min(scale_width, scale_height)
    window_width = int(img.shape[1] * scale)
    window_height = int(img.shape[0] * scale)

    cv2.namedWindow('dst_rt', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('dst_rt', window_width, window_height)

    cv2.imshow('dst_rt', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def loadModel(model_path, weight_path):
    json_file = open(model_path, 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    # load weights into new model
    model.load_weights(weight_path)
    print("Loaded model from disk")
    # evaluate loaded model on test data
    model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

def predictImage(img, model):
    img = np.dot(np.array(img, dtype='float32'), [[0.2989], [0.5870], [0.1140]]) / 255
    x = keras_image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    classes = model.predict_classes(images, batch_size=10)
    return classes[0][0]

def predictFacialLandmark(img, detector):
    img = imutils.resize(img, width=500)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect faces in the grayscale image
    rects = detector(gray, 1)
    return rects

def readImage(img_path):
    image = cv2.imread(img_path)
    image = imutils.resize(image, width=500)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image, gray

def drawEyes(eye, image):
    (x, y, w, h) = cv2.boundingRect(np.array([eye]))
    h = w
    y = y - h / 2
    print x, y, w, h
    roi = image[y:y + h, x:x + w]
    roi = imutils.resize(roi, width=24, inter=cv2.INTER_CUBIC)
    print roi.shape
    return roi

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-img", "--image", required=True,
                help="path to test image")
args = vars(ap.parse_args())

if __name__ == "__main__":
    # Define counter
    COUNTER = 0
    ALARM_ON = False
    MAX_FRAME = 20
    # Load model
    model = loadModel('trained_model/model_1496828307.9.json', "trained_model/weight_1496828307.9.h5")
    img_path = args['image']
    img = cv2.imread(img_path)
    image, gray = readImage(img_path=img_path)

    # Predict Facial Landmark
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('trained_data.dat')
    rects = predictFacialLandmark(img=img, detector=detector)

    # grab the indexes of the facial landmarks for the left and
    # right eye, respectively
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        leftEyeRatio = shape[lStart:lEnd]
        leftEye = drawEyes(leftEyeRatio, image)
        # # showImage(leftEye, 450, 450)
        classLeft = predictImage(leftEye, model=model)
        print classLeft
        rightEyeRatio = shape[rStart:rEnd]
        rightEye = drawEyes(rightEyeRatio, image)
        classRight = predictImage(rightEye, model=model)
        print classRight


