## Scattering_maxp

### Introduction
> This is for the scattering-maxp network, which is the modified version of scattering network by S.Mallat. In paper [1], a way of combining scattering network with max-pooling network is introduced. This repository is for support of experiments in [1]. The experiments are image classifications tasks. The training data sets we use in the experiments are Caltech-101 [2] and Caltech-256 [3]. Special thanks to [4] for the base codes.


### Set-up 
> The experiments requires TensorFlow 1.15.1 & Keras 2.2.7-tf. You can download the image data sets at http://www.vision.caltech.edu/Image_Datasets/Caltech101/#Download
and http://www.vision.caltech.edu/Image_Datasets/Caltech256/. We prepare three scattering based models, namned scattering, scattering-maxp and scattering-naivep. (More details are in [1]).

![experiment_model_1](https://user-images.githubusercontent.com/55676509/100437400-0802a580-30e4-11eb-821d-e6fd223821a9.png)

![experiment_model_2](https://user-images.githubusercontent.com/55676509/100437431-1650c180-30e4-11eb-8a1a-a4957d9ba7bc.png)

![experiment_model_3](https://user-images.githubusercontent.com/55676509/100437445-19e44880-30e4-11eb-9fb0-4ae145a6cbd8.png)




### Main 1
> Model 1: Scattering-maxp of Caltech-101.

```python
from PIL import Image
import os, glob

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf #version: 1.15.1

from Scattering_maxp.keras import Scattering2D as Scattering_maxp

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Flatten

# mdoel construction
tf.reset_default_graph()
inputs_1 = Input(shape=(224, 224))
x1 = Scattering_maxp(J = 3, L = 8)(inputs_1)
x1 = Dense(512, activation ='relu')(x1)
x1 = Dense(512, activation ='relu')(x1)
x1 = Dense(256, activation ='relu')(x1)
x1 = Dense(256, activation ='relu')(x1)
output_1 = Dense(102, activation = 'softmax')(x1)

model1 = Model(inputs_1, output_1)
model1.summary()
model1.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# data preprocessing
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

#training session
with tf.device('/gpu:0'):
    model1_hist = model1.fit(X_train, Y_train, 
                          validation_data = (X_test,Y_test), epochs=300, batch_size=256)

```




### Main 2
> Model 2: Scattering-maxp for Caltech-256.


```python
from PIL import Image
import os, glob

from sklearn.model_selection import train_test_split
import numpy as np
import tensorflow as tf #version: 1.15.1
from tensorflow.keras.layers import Input, Dense, Flatten 
from tensorflow.keras.models import Model
  
from Scattering_maxp.keras import Scattering2D as Scattering_maxp
  
  # model construction
tf.reset_default_graph()
inputs_2 = Input(shape=(224,224))
x2 = Scattering_maxp(J = 3, L = 8)(inputs_2)
x2 = Dense(512, activation ='relu')(x2)
x2 = Dense(512, activation ='relu')(x2)
x2 = Dense(256, activation ='relu')(x2)
x2 = Dense(256, activation ='relu')(x2)
output_2 = Dense(257, activation = 'softmax')(x2)
  
model2 = Model(inputs_2, output_2)
model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model2.summary()
 
  # data preprocessing
caltech_dir = "./datasets/256_ObjectCategories"
classes = os.listdir(caltech_dir)
nb_classes = len(classes)
  
X = []; Y = []

image_w = 224; image_h = 224;
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
    
# training session
def scheduler(epochs):
    if epochs < 50 :
        return 0.001
    elif (epochs > 51) & (epochs < 75):
        return 0.0001
    else :
        return 0.00001

callback = tf.keras.callbacks.LearningRateScheduler(scheduler)

with tf.device('/gpu:0'):
    model2_hist = model2.fit(X_train, Y_train, 
                        validation_data = (X_test,Y_test), callbacks =[callback], epochs=200, batch_size=256)
```



[1] T. Ki and Y. Hur, "Deep Scattering Network with Max-pooling", To appear (as a poster) in Proceedings of 2021 DCC, IEEE Computer Society Conference Publishing Services.

[2] T. Ki and Y. Hur, "Deep Scattering Network with Max-pooling", preprint, 2021, https://arxiv.org/abs/2101.02321.

[3] L. Fei-Fei, R. Fergus, and P. Perona, “Learning Generative Visual Models from few
Training Examples: An Incremental Bayesian Approach Tested on 101 Object Categories,”
in Conference on Computer Vision and Pattern Recognition Workshop, 2004.

[4] G. Griffin, A. Holub, and P. Perona, “Caltech-256 Object Category Dataset,” preprint,
2007.

[5] M. Andreux, T. Angles, G. Exarchakis, R. Leonarduzzi, G. Rochette, L. Thiry, J. Zarka, S. Mallat, J. And{\'e}n, E. Belilovsky, et.al., "Kymatio: Scattering Transforms in {P}ython", Journal of Machine Learning Research", 21, 1 - 6, 2020.
