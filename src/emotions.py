import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D,MaxPooling2D,Dense,Dropout,Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
import argparse


USE_WEBCAM= True  # If true then it will use the webcam,untill it is False will detect the emotions in a video file chosen.

# Command Line Argument

ap = argparse.ArgumentParser()
ap.add_argument("--run",help="train/test/picture")
run = ap.parse_args().run

#data generators
train_dir = 'data/train'
test_dir = 'data/test'

train_num = 28709
test_num = 7178
batch_size = 64
epochs = 50

train_datagen = ImageDataGenerator(rescale=1./255)
val_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')

validation_generator = val_datagen.flow_from_directory(
        test_dir,
        target_size=(48,48),
        batch_size=batch_size,
        color_mode="grayscale",
        class_mode='categorical')


# The model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48,48,1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

print(model.summary())
# If you want to train the same model
if run == "train":
    model.compile(loss='categorical_crossentropy',optimizer=Adam(lr=0.0001, decay=1e-6),metrics=['accuracy'])
    model_info = model.fit_generator(
            train_generator,
            steps_per_epoch=train_num // batch_size,
            epochs=epochs,
            validation_data=validation_generator,
            validation_steps=test_num // batch_size)
   
    model.save_weights('model_2.h5')


elif run =="picture":
    model.load_weights('model.h5')

   
    cv2.ocl.setUseOpenCL(False)
   
    emotion_dict = {0: "ANGRY", 1: "DISGUSTED", 2: "FEARFUL", 3: "HAPPY", 4: "NEUTRAL", 5: "SAD", 6: "SURPRISED"}
    

    imagepath = "./test/friends.jpg"
    
    img = cv2.imread(imagepath,0)

    print(img)

    while True:

        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

        faces = facecasc.detectMultiScale(img,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y-90), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = img[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            # print(prediction)
            maxindex = int(np.argmax(prediction))
            # print(maxindex)
            cv2.putText(img, emotion_dict[maxindex], (x+20, y-80), cv2.FONT_HERSHEY_DUPLEX, 3, (0, 0, 0), 4, cv2.LINE_AA)

    # cv2.imshow('Video', cv2.resize(img,(1400,1200),interpolation = cv2.INTER_CUBIC))
    #     if cv2.waitKey(1) & 0xFF == ord('q'):
    #         break

        cv2.imshow('Picture', cv2.resize(img,(1400,1200),interpolation = cv2.INTER_CUBIC))
        
    
    cv2.destroyAllWindows()




# emotions will be displayed on your face from the webcam feed
elif run == "test":
    model.load_weights('model.h5')

    # prevents openCL usage and unnecessary logging messages
    cv2.ocl.setUseOpenCL(False)
 
    emotion_dict = {0: "ANGRY", 1: "DISGUSTED", 2: "FEARFUL", 3: "HAPPY", 4: "NEUTRAL", 5: "SAD", 6: "SURPRISED"}
    
    
    cap = None
    if (USE_WEBCAM == True):
       cap = cv2.VideoCapture(0)
    else:
       cap = cv2.VideoCapture('./test/testvdo.mp4')
       
    while True:
        # Find haar cascade to draw bounding box around face
        ret, frame = cap.read()
        if not ret:
            break
        facecasc = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = facecasc.detectMultiScale(gray,scaleFactor=1.3, minNeighbors=5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y-30), (x+w, y+h+10), (255, 0, 0), 2)
            roi_gray = gray[y:y + h, x:x + w]
            cropped_img = np.expand_dims(np.expand_dims(cv2.resize(roi_gray, (48, 48)), -1), 0)
            prediction = model.predict(cropped_img)
            # print(prediction)
            maxindex = int(np.argmax(prediction))
            # print(maxindex)
            cv2.putText(frame, emotion_dict[maxindex], (x+20, y-40), cv2. FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

        cv2.imshow('Video', cv2.resize(frame,(1400,1200),interpolation = cv2.INTER_CUBIC))
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

