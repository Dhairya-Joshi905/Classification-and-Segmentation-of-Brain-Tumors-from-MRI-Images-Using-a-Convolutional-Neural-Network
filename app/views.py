from django.shortcuts import render
import tensorflow as tf
from django.core.files.storage import FileSystemStorage

import os
from tensorflow.python.ops import image_ops
from tensorflow.python.ops import io_ops
import numpy as np
import keras

import matplotlib.pyplot as plt
# Create your views here.
BASE_DIR = os.path.dirname(os.path.dirname(__file__))
#model_path = BASE_DIR + '/model'
model_classifier = BASE_DIR+"/model/2019-06-07_VGG_model.h5s"
model_seg = BASE_DIR+"/model/BraTs2020.h5"
#model = tf.keras.models.load_model(model_path)

import keras
from keras.models import Model,load_model,Sequential
from keras.callbacks import EarlyStopping,ModelCheckpoint
from keras.layers.merge import concatenate

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input ,BatchNormalization , Activation 
from keras.layers.convolutional import Conv2D, UpSampling2D
from keras.layers import Dense,Conv2D,MaxPooling2D,BatchNormalization,Flatten,Dropout
model_path = "D:/Notebooks/Brain_Tumor/checkpoint/2019-06-07_VGG_model.h5"

def Convolution(input_tensor,filters):
    
    x = Conv2D(filters=filters,kernel_size=(3, 3),padding = 'same',strides=(1, 1))(input_tensor)
    x = BatchNormalization()(x)
    x = Activation('relu')(x) 
    return x


def model(input_shape):
    
    inputs = Input((input_shape))
    
    conv_1 = Convolution(inputs,32)
    maxp_1 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_1)
    
    conv_2 = Convolution(maxp_1,64)
    maxp_2 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_2)
    
    conv_3 = Convolution(maxp_2,128)
    maxp_3 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_3)
    
    conv_4 = Convolution(maxp_3,256)
    maxp_4 = MaxPooling2D(pool_size = (2, 2), strides = (2, 2), padding = 'same') (conv_4)
    
    conv_5 = Convolution(maxp_4,512)
    upsample_6 = UpSampling2D((2, 2)) (conv_5)
    
    conv_6 = Convolution(upsample_6,256)
    upsample_7 = UpSampling2D((2, 2)) (conv_6)
    
    upsample_7 = concatenate([upsample_7, conv_3])
    
    conv_7 = Convolution(upsample_7,128)
    upsample_8 = UpSampling2D((2, 2)) (conv_7)
    
    conv_8 = Convolution(upsample_8,64)
    upsample_9 = UpSampling2D((2, 2)) (conv_8)
    
    upsample_9 = concatenate([upsample_9, conv_1])
    
    conv_9 = Convolution(upsample_9,32)
    outputs = Conv2D(1, (1, 1), activation='sigmoid') (conv_9)
    
    model = Model(inputs=[inputs], outputs=[outputs]) 
    
    return model

model = model(input_shape = (240,240,1))

reconstructed_model = keras.models.load_model(model_path)
def index(request):

    return render(request, "app/index.html")


def predic(request):

    image = request.FILES['image-file']
    fs = FileSystemStorage()
    imagepath = fs.save(image.name, image)
    imagepath = fs.url(imagepath)
    print("---------------------------------------------------")
    print(imagepath)
    
    test_image = '.'+imagepath
    
    #img = tf.keras.preprocessing.image.load_img(test_image)
    #x = tf.keras.preprocessing.image.img_to_array(img)
    #x = tf.data.Dataset.from_tensors(x)
    img_class = io_ops.read_file(test_image)
    img_class = image_ops.decode_image(
        img_class, channels=3, expand_animations=False)
    img_class = image_ops.resize_images_v2(img_class, (224,224), method='bilinear')
    img_class.set_shape((224, 224, 3))
    x = tf.data.Dataset.from_tensors(img_class)
    x = x.batch(1)

    img_seg = io_ops.read_file(test_image)
    img_seg = image_ops.decode_image(
        img_seg, channels=1, expand_animations=False)
    img_seg = image_ops.resize_images_v2(img_seg, (240,240), method='bilinear')
    img_seg.set_shape((240, 240, 1))
    y = tf.data.Dataset.from_tensors(img_seg)
    y = y.batch(1)

    #x = x.batch(1)
    #data = next(iter(x))

    #img_path = os.path.join(BASE_DIR, 'media/photo')
    #img_path = os.path.dirname(img_path)
    # test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    #    img_path, color_mode='rgb', batch_size=1)

    result = reconstructed_model.predict(x)
    print("---------------------------")
    print(result)
    model.load_weights(model_seg)
    p=model.predict(y)
    imm = plt.imshow(p[0])
    plt.savefig(BASE_DIR+"/media/photo/plot.png")
    result_ph = np.asarray(result[0][0])
    

    contex = {
        'result_ph': result_ph,
       

    }
    return render(request, "app/predict.html", context=contex)
