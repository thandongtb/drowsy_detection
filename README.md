# Framgia Drowsy Detection with OpenCV and CNN
Drowsy detection using OpenCV and Convolution Neural Networks.
## Requirement
*  Python 2.7
*  OpenCV
*  Keras
*  Tensorflow
*  Scipy
## Download the dataset
Please download the [dataset](http://parnec.nuaa.edu.cn/xtan/data/ClosedEyeDatabases.html) and unzip to the ***/dataset/*** folder. We will have four folders:
* closedLeftEyes
* closedRightEyes
* openLeftEyes
* openRightEyes

Please download [this file](https://github.com/AKSHAYUBHAT/TensorFace/blob/master/openface/models/dlib/shape_predictor_68_face_landmarks.dat) and place into the current folder. Rename it with ***trained_data.dat***

You maybe collect any different data for testing the algorithms
## Traning new data
```
    python eye_preprocess.py
    python eye_training_cnn.py
```

## Testing with local image
~~~
python predict_eye.py -img path_to_local_image
~~~

## Testing Real-time with Camera
Firstly, make sure the camera is usable in your computer. Run this application real time with this command:
~~~
python drowsy_predict.py
~~~

***Press Q*** to quit the application
