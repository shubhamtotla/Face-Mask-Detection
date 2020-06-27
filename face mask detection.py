#FACE MASK DETECTION

#TRAINING DATASET USING CNN

import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
import os
from os import listdir

print("The number of images with facemask labelled 'yes':",len(os.listdir('with_mask')))
print("The number of images with facemask labelled 'no':",len(os.listdir('without_mask')))

'''We augment our dataset to include more number of images for our training. 
In this step of data augmentation, we rotate and flip each of the images in our dataset'''

from skimage import io 
from skimage.transform import rotate, AffineTransform, warp
import matplotlib.pyplot as plt
import random
from skimage import img_as_ubyte
from skimage.util import random_noise
import numpy as np
import cv2

#Define functions for data augmentation

def normal(image):
    return (image)

def anticlockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, angle)

def clockwise_rotation(image):
    angle= random.randint(0,180)
    return rotate(image, -angle)

def h_flip(image):
    return  np.fliplr(image)

#We do not need vertical flip

'''def v_flip(image):
    return np.flipud(image)'''

def add_noise(image):
    return random_noise(image)

def blur_image(image):
    return cv2.GaussianBlur(image, (9,9),0)


transformations = {'rotate anticlockwise': anticlockwise_rotation,
                      'normal': normal,
                      'rotate clockwise': clockwise_rotation,
                      'horizontal flip': h_flip, 
                      'adding noise': add_noise,
                   'blurring image':blur_image
                 }                #use dictionary to store names of functions 


images_path="with_mask" #path to original images
augmented_path="C:/Users/SHUBHAM TOTLA/Desktop/observations-master/experiements/data/augmented_with_image"
os.mkdir(augmented_path) 
images=[] # to store paths of images from folder

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

images_to_generate=1380  #you can change this value according to your requirement
i=1                        # variable to iterate till images_to_generate

while i<=images_to_generate:    
    image=random.choice(images)
    original_image = io.imread(image)
    transformed_image=None
    print(i)
    n = 0       #variable to iterate till number of transformation to apply
    transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image
    
    while n <= transformation_count:
        key = random.choice(list(transformations)) #randomly choosing method to call
        transformed_image = transformations[key](original_image)
        n = n + 1
        
    new_image_path= "%s/augmented_image_%s.jpg" %(augmented_path, i)
    transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].
    transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
    cv2.imwrite(os.path.join(augmented_path , new_image_path), transformed_image) # save transformed image to path
    i =i+1

#SIMILARLY FOR WITHOUT MASK

images_path="without_mask" #path to original images
augmented_path="C:/Users/SHUBHAM TOTLA/Desktop/observations-master/experiements/data/augmented_without_image"
os.mkdir(augmented_path) 
images=[] # to store paths of images from folder

for im in os.listdir(images_path):  # read image name from folder and append its path into "images" array     
    images.append(os.path.join(images_path,im))

images_to_generate=1370  #you can change this value according to your requirement
i=1                        # variable to iterate till images_to_generate

while i<=images_to_generate:    
    image=random.choice(images)
    original_image = io.imread(image)
    transformed_image=None
    print(i)
    n = 0       #variable to iterate till number of transformation to apply
    transformation_count = random.randint(1, len(transformations)) #choose random number of transformation to apply on the image
    
    while n <= transformation_count:
        key = random.choice(list(transformations)) #randomly choosing method to call
        transformed_image = transformations[key](original_image)
        n = n + 1
        
    new_image_path= "%s/augmented_image_%s.jpg" %(augmented_path, i)
    transformed_image = img_as_ubyte(transformed_image)  #Convert an image to unsigned byte format, with values in [0, 255].
    transformed_image=cv2.cvtColor(transformed_image, cv2.COLOR_BGR2RGB) #convert image to RGB before saving it
    cv2.imwrite(os.path.join(augmented_path , new_image_path), transformed_image) # save transformed image to path
    i =i+1
    
#SPLITTING DATASET
import shutil
import random

# # Creating Train / Val / Test folders (One time use)

root_dir = 'C:/Users/SHUBHAM TOTLA/Desktop/observations-master/experiements/data'
classes_dir = ['/augmented_with_image', '/augmented_without_image']

val_ratio = 0.15
test_ratio = 0.05

for cls in classes_dir:
    os.makedirs(root_dir +'/train' + cls)
    os.makedirs(root_dir +'/val' + cls)
    os.makedirs(root_dir +'/test' + cls)


    # Creating partitions of the data after shuffeling
    src = root_dir + cls # Folder to copy images from

    allFileNames = os.listdir(src)
    np.random.shuffle(allFileNames)
    train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames),
                                                              [int(len(allFileNames)* (1 - val_ratio + test_ratio)), 
                                                               int(len(allFileNames)* (1 - test_ratio))])


    train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
    val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
    test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

    print('Total images: ', len(allFileNames))
    print('Training: ', len(train_FileNames))
    print('Validation: ', len(val_FileNames))
    print('Testing: ', len(test_FileNames))

    # Copy-pasting images
    for name in train_FileNames:
        shutil.copy(name, root_dir +'/train' + cls)

    for name in val_FileNames:
        shutil.copy(name, root_dir +'/val' + cls)

    for name in test_FileNames:
        shutil.copy(name, root_dir +'/test' + cls)
        
#COPY VAL IMAGES TO TEST

#CREATE CNN MODEL

# Initialising the CNN

classifier = Sequential()
# Step 1 - Convolution
classifier.add(Conv2D(100, (3, 3), input_shape = (150, 150, 3), activation = 'relu'))

# Step 2 - Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Adding a second convolutional layer
classifier.add(Conv2D(100, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Step 3 - Flattening
classifier.add(Flatten())
classifier.add(Dropout(0.5))

# Step 4 - Full connection
classifier.add(Dense(units = 50, activation = 'relu'))
classifier.add(Dense(units = 1, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1.0/255,
                                   rotation_range=40,
                                   width_shift_range=0.2,
                                   height_shift_range=0.2,
                                   shear_range=0.2,
                                   zoom_range=0.2,
                                   horizontal_flip=True,
                                   fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
        'C:/Users/SHUBHAM TOTLA/Desktop/observations-master/experiements/data/train',
        target_size=(150,150),#Should be same as expected above
        batch_size=32,
        class_mode='binary')

test = test_datagen.flow_from_directory(
        'C:/Users/SHUBHAM TOTLA/Desktop/observations-master/experiements/data/test',
        target_size=(150, 150),
        batch_size=32,
        class_mode='binary')

from keras.callbacks import ModelCheckpoint
checkpoint = ModelCheckpoint('model-{epoch:03d}.model',monitor='val_loss',verbose=0,save_best_only=True,mode='auto')
classifier.fit_generator(
        train,
        steps_per_epoch=2475,#No. of images we have in training set
        epochs=30,
        validation_data=test,
        validation_steps=275,
        callbacks = [checkpoint])#No. of images we have in test set

#CHANGE LEARNING RATE TO LR FOR NEW KERAS VERSION
'''import h5py
f = h5py.File('model-facemask.h5','r+')
data_p = f.attrs['training_config']
data_p = data_p.decode().replace("learning_rate","lr").encode()
f.attrs['training_config'] = data_p
f.close()'''

from keras.models import load_model

model = load_model('model-facemask.h5')
model.summary()
import numpy as np
#FACE MASK DETECCTION USING OPENCV2
import cv2
classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
webcam = cv2.VideoCapture(0) #Use camera 0
size=4
while True:
    (rval, im) = webcam.read()
    im=cv2.flip(im,1,1) #Flip to act as a mirror

    # Resize the image to speed up detection
    mini = cv2.resize(im, (im.shape[1] // size, im.shape[0] // size))
    # detect MultiScale / faces 
    faces = classifier.detectMultiScale(mini)

    # Draw rectangles around each face
    for f in faces:
        (x, y, w, h) = [v * size for v in f] #Scale the shapesize backup
        #Save just the rectangle faces in SubRecFaces
        face_img = im[y:y+h, x:x+w]
        resized=cv2.resize(face_img,(224,224))
        normalized=resized/255.0
        reshaped=np.reshape(normalized,(1,224,224,3))
        reshaped = np.vstack([reshaped])
        (mask, withoutMask)=model.predict(reshaped)[0]
       # determine the class label and color we'll use to draw
		# the bounding box and text
        label = "Mask" if mask > withoutMask else "No Mask"
        print(label)
        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
        cv2.putText(im, label, (x, y - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(im, (x,y),(x+w,y+h), color, 2)
        
    # Show the image
    cv2.imshow('LIVE',   im)
    key = cv2.waitKey(1)
    # if Esc key is press then break out of the loop 
    if key == 27: #The Esc key
        break
# Stop video
webcam.release()

# Close all started windows
cv2.destroyAllWindows()