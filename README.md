## Scattering_maxp

### Introduction
> This is for the scattering-maxp network, which is the modified version of scattering network by S.Mallat. In paper [1], a way of combining scattering network with max-pooling network is introduced. This repository is for support of experiments in [1]. The experiments are image classifications tasks. Where the training data sets are Caltech-101 [2] and Caltech-256 [3]. 


### Set-up 
> The experiments requires TensorFlow 1.15.1 & Keras 2.2.7-tf. You can download the image data sets at http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download
and http://www.vision.caltech.edu/Image_Datasets/Caltech256/. We prepare three scattering based models, namned scattering, scattering-maxp and scattering-naivep. (More details are in [1]).

![experiment_model_1](https://user-images.githubusercontent.com/55676509/100437400-0802a580-30e4-11eb-821d-e6fd223821a9.png)

![experiment_model_2](https://user-images.githubusercontent.com/55676509/100437431-1650c180-30e4-11eb-8a1a-a4957d9ba7bc.png)

![experiment_model_3](https://user-images.githubusercontent.com/55676509/100437445-19e44880-30e4-11eb-9fb0-4ae145a6cbd8.png)


### Main 
> Model 2: scattering-maxp for Caltech-256.


```python
  
  from PIL import Image
  import os, glob
  import pandas as pd

  from sklearn.model_selection import train_test_split
  import numpy as np
  import tensorflow as tf #version: 1.15.1
  from tensorflow.keras.layers import Input, Dense, Flatten 
  from tensorflow.keras.models import Model
  
  from Scattering_maxp.keras import Scattering2D as Scattering_maxp
  
  inputs_2 = Input(shape=(224,224))
  x = Scattering_maxp(J =3, L = 8)(inputs_1)
  x = Dense(512, activation ='relu')(x)
  x = Dense(512, activation ='relu')(x)
  x = Dense(256, activation ='relu')(x)
  x = Dense(256, activation ='relu')(x)
  output_2 = Dense(257, activation = 'softmax')(x)
  
  model2 = Model(inputs_2, output_2)
  model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
  model2.summary()
  
  model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

  caltech_dir = "./datasets/101_ObjectCategories"
  classes = os.listdir(caltech_dir)
  nb_classes = len(classes)
  
  
  X = []; Y = []
  
  image_w = 224; image_h = 224
  
  for idx, f in enumerate(classes):
    label = [ 0 for i in range(nb_classes) ]
    label[idx] = 1
    image_dir = caltech_dir + "/" + f
    files = glob.glob(image_dir + "/*.jpg")
    for i, fname in enumerate(files):
        img = Image.open(fname)
        img = img.convert("L")
        img = img.resize((image_w, image_h))        
        data = np.asarray(img)
        
        X.append(data); Y.append(label)
        for ang in range(-20,20,5):
            img2= img.rotate(ang); data = np.asarray(img2)
            X.append(data); Y.append(label)
            img2 = img2.transpose(Image.FLIP_LEFT_RIGHT); data = np.asarray(img2)
            X.append(data); Y.append(label)
            
            
    X = np.array(X); Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y)
    XY_val = (X_train, X_test, Y_train, Y_test)
    
  
   with tf.device('/gpu:0'):
     model2_hist = model2.fit(X_train, Y_train, validation_data = (X_test,Y_test), callbacks =[callback], epochs=300, batch_size=256)






```



[1] T. Ki and Y. Hur, "Deep scattering transform with Max-pooling", submitted.

[2] L. Fei-Fei, R. Fergus, and P. Perona, “Learning generative visual models from few
training examples: An incremental Bayesian approach tested on 101 object categories,”
in Conference on Computer Vision and Pattern Recognition Workshop, 2004.

[3] G. Griffin, A. Holub, and P. Perona, “Caltech-256 object category dataset,” preprint,
2007.
