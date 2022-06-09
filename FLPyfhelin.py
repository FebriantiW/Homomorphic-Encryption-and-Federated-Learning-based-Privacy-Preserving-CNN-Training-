import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import img_to_array
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
#from tensorflow.keras.applications import VGG16,VGG19,ResNet50,DenseNet121,InceptionV3
from tensorflow.keras.models import Model
from keras.models import load_model
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv2D,Flatten,MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping,ReduceLROnPlateau
from keras.layers import Conv2D, SeparableConv2D, MaxPool2D, LeakyReLU, Activation
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization,AveragePooling2D
import os
from os.path import exists
from IPython.display import clear_output
from kerastuner import HyperModel
from Pyfhel import Pyfhel, PyPtxt, PyCtxt
import pickle
import time

INIT_LR = 1e-3
EPOCHS = 10
BS = 32
SCALE= 1
input_shape  = (int(256*SCALE), int(256*SCALE), 3)
image_size  = (int(256*SCALE), int(256*SCALE))

def prep_df(folder, shuffle=True):
    dataframe = []
    subdir = [sub for sub in os.scandir(folder)]
    for sub in subdir:
        if sub.is_dir() or sub.is_file():
            path = sub.path
            list_files = os.listdir(path)
            #list_files=[os.path.join(path,x) for x in list_files]
            list_files=[os.path.join(os.path.abspath(path),x) for x in list_files]
            entry = [[file,sub.name] for file in list_files]
            dataframe.extend(entry)
            
    dframe =pd.DataFrame(dataframe, columns =['Path', 'Label'])
    
    if shuffle == True:
        dframe = dframe.sample(frac=1).reset_index(drop=True)
        
    return dframe

def get_test_data(df_test,test_path):
    
    test_data_gen = ImageDataGenerator(rescale = 1./255)
    test = test_data_gen.flow_from_dataframe(
      dataframe = df_test,
      directory = test_path,
      x_col='Path', 
      y_col='Label',
      target_size=image_size,
      #color_mode='grayscale',
      shuffle=False, 
      class_mode='categorical',
      batch_size=BS,
      )
    return test

def get_train_data(df_train,train_path,index, num_client):
    
    ratio = int(len(df_train.index)/num_client)
    start = index * ratio 
    end = start + ratio
    df_train = df_train[start:end] 
   
    image_gen = ImageDataGenerator(
                rescale = 1./255,
                shear_range = 0.2,
                zoom_range = 0.2,
                horizontal_flip = True,
                validation_split = 0.1,
    )

    train = image_gen.flow_from_dataframe(
      dataframe = df_train,
      directory = train_path,
      x_col='Path', 
      y_col='Label',
      subset='training',
      target_size=image_size,
      #color_mode='grayscale'        
      class_mode='categorical',
      shuffle=True, 
      batch_size=BS,
      )
    
    val = image_gen.flow_from_dataframe(
      dataframe = df_train,
      directory = train_path,
      x_col='Path', 
      y_col='Label',
      subset='validation',
      target_size=image_size,
      #color_mode='grayscale'        
      class_mode='categorical',
      shuffle=True, 
      batch_size=BS,
      )

    return train,val



def create_model(load_model_path=None):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(32, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(64, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))
    model.add(Conv2D(128, (3, 3), activation="relu", input_shape=input_shape))
    model.add(MaxPooling2D(pool_size = (2, 2)))

    model.add(Flatten())
    model.add(Dense(activation = 'relu', units = 128))
    model.add(Dense(activation = 'relu', units = 64))
    model.add(Dense(activation = 'softmax', units = 2))
    #cnn.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
                
    #model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    opt = Adam(learning_rate=INIT_LR, decay=INIT_LR /10)
    model.compile(loss="categorical_crossentropy", optimizer=opt,metrics=["accuracy"])
    
    if load_model_path:
        model=load_model(load_model_path)

    return model

    
def save_weights(model,ind):
    #names = [weight.name for layer in model.layers for weight in layer.weights]
    weights = np.array(model.get_weights(),dtype='object')
    np.save('weights/weights' + ind+ '.npy', weights, allow_pickle=True)
    return

def load_weights(ind):
    weights=np.load('weights/weights' + ind + '.npy',allow_pickle=True)
    model=create_model()
    model.set_weights(weights)
    return model

def train_server(train_ds,val_ds, epoch=10):
    model=create_model()
    
    early = EarlyStopping(monitor="loss", mode="min", patience=3)
    lr_red = ReduceLROnPlateau(monitor="loss", patience = 2, verbose=1,factor=0.3, min_lr=0.000001)

    checkpoint_path = "weights/main.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    checkpoint =  ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,save_best_only=True, verbose=1,monitor='accuracy', mode='auto')

    callbacks_list = [early, lr_red,checkpoint]
    model.fit(train_ds, validation_data=val_ds, epochs=epoch, callbacks=callbacks_list)
    model.load_weights(checkpoint_path)
    save_weights(model,'main')
    model.save('main_model.hdf5')

    return

def train_clients(dataframe, train_path, num_clients, epoch=10):
    model=create_model('main_model.hdf5')
    
    #dataframe = prep_df(train_path,shuffle=True):
    
    for i in range(num_clients):       
        train_ds, val_ds = get_train_data(dataframe, train_path, i, num_clients)
        early = EarlyStopping(monitor="loss", mode="min", patience=5,restore_best_weights=True)
        lr_red = ReduceLROnPlateau(monitor="loss", patience = 2, verbose=1,factor=0.3, min_lr=0.000001)
        
        checkpoint_path = "weights/client_" + str(i+1)+ ".ckpt"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        checkpoint =  ModelCheckpoint(filepath=checkpoint_path,save_weights_only=True,save_best_only=True, verbose=1,monitor='accuracy', mode='auto')
        #model.fit(train_ds, validation_data=val_ds, callbacks=[checkpoint], epochs=EPOCHS)
        model.fit(train_ds, validation_data=val_ds, callbacks=[checkpoint,early,lr_red], epochs=epoch)
        #model.load_weights(checkpoint_path)
        
        save_weights(model,str(i+1))
        
    return

def encrypt_export_weights(indx):
    HE = get_pk()
    model = load_weights(str(indx+1))
    start = time.time()
    encrypted_weights={}
    for i in range(len(model.layers)):
        if model.layers[i].get_weights()!=[]:
            encrypted =[]
            weights = model.layers[i].get_weights()   

            for j in range(len(weights)):
                    weight = weights[j]
                    shape = weight.shape
                    weight = weight.flatten()
                    array= np.empty(len(weight),dtype=PyCtxt)
                    
                    for k in np.arange(len(weight)):
                        array[k] = HE.encryptFrac(weight[k])
                    
                    enc_array = array.reshape(shape)
                    #enc_array = np.array(enc_array,dtype=PyCtxt)
                    encrypted_weights['c_'+str(i)+'_'+str(j)]=enc_array
    
    end = time.time()
    print('Time to encrypt weights:',end-start)
    filename =  "weights/client_" + str(indx+1)+ ".pickle"
    export_weights(filename, encrypted_weights)
    
    return

def export_weights(filename, encrypted_weights):
    HE = get_pk()
    dic = {}
    dic['key']=HE
    dic['val']=encrypted_weights
    start = time.time()
    with open(filename, 'wb') as handle:
        pickle.dump(dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
    end = time.time()
    print('Time to export weights to pickle:',end-start)
    return

def export_encrypted_clients_weights(num_client):
    start = time.time()
    for i in range(num_client):
        encrypt_export_weights(i)
        print('Weights exported: Client',i+1)
    end = time.time()
    print('Total time to encrypt and export:',end-start)
    return

def get_sk():
    filename =  "privatekey.pickle"
    with open(filename, 'rb') as handle:
            key = pickle.load(handle)
    
    HE = key['HE']
    HE.from_bytes_context(key['con'])
    HE.from_bytes_publicKey(key['pk'])
    HE.from_bytes_secretKey(key['sk'])
    
    return HE

def decrypt_import_weights(filename):
    start = time.time()
    dec_weights=decrypt_weights(filename)
    end = time.time()
    print('Time to decrypt:',end-start)
    
    model=create_model('main_model.hdf5')
    model.get_weights()
    for i in range(len(model.layers)):
        if model.layers[i].get_weights()!=[]: 
            weights = model.layers[i].get_weights()
            weight=[]
            for j in range(len(weights)):
                weight.append(dec_weights['c_'+str(i)+'_'+str(j)])
            
            model.layers[i].set_weights(weight)
            
    model.save('agg_model.hdf5')
    return model

def decrypt_weights(filename):
    HE = get_sk()
    enc_weights={}
    dec_weights={}
    enc_weights=import_encrypted_weights(filename)
    
    for key in enc_weights:
        arr = enc_weights[key]
        shape = arr.shape
        weight = arr.flatten()
        
        for l in range(len(weight)):
            weight[l]= HE.decryptFrac(weight[l])

        dec_array = weight.reshape(shape)
        dec_weights[key] = dec_array

    return dec_weights


def import_encrypted_weights(filename):
    weights={}
    model = create_model()
    start = time.time()
    #filename =  "weights/client_" + str(indx+1)+ ".pickle"
    with open(filename, 'rb') as handle:
        dct = pickle.load(handle)

    cweights=dct['val']
    HE2 = dct['key']
    enc_weights={}

    for key in cweights:
        arr = cweights[key]
        shape = arr.shape
        weight = arr.flatten()

        for l in np.arange(len(weight)):
            weight[l]._pyfhel = HE2

        enc_array = weight.reshape(shape)
        enc_weights[key] = enc_array
    
    end = time.time()
    print('Time to import:',end-start)    
    return enc_weights

def gen_pk(s=128, m=2048):
    HE = Pyfhel()  
    HE.contextGen(p=65537, sec = s, m=m)
    HE.keyGen()
    
    keys ={}
    keys['HE'] = HE
    keys['con'] = HE.to_bytes_context()
    keys['pk'] = HE.to_bytes_publicKey()
    
    filename =  "publickey.pickle"
    with open(filename, 'wb') as handle:
        pickle.dump(keys, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return HE

def get_pk():
    filename =  "publickey.pickle"
    with open(filename, 'rb') as handle:
        key = pickle.load(handle)

    HE2 = key['HE']
    HE2.from_bytes_context(key['con'])
    HE2.from_bytes_publicKey(key['pk'])

    return HE2

def gen_rekey():
    filename =  "publickey.pickle"
    with open(filename, 'rb') as handle:
        key = pickle.load(handle)
        
    relinKeySize=5
    HE.relinKeyGen(bitCount=1, size=relinKeySize)
    return HE

def aggregate_encrypted_weights(num_client):
    dct_weights ={}
    denom = float(1/num_client)
    start = time.time()
    HE = get_pk()
    c_denom = HE.encryptFrac(denom)
    for i in range(num_client):
        enc_weights={}
        filename =  "weights/client_" + str(i+1)+ ".pickle"
        enc_weights =import_encrypted_weights(filename)
        
        for key in enc_weights:
            if i == 0:
                arr = enc_weights[key]
                dct_weights[key] = np.zeros_like(arr,dtype=PyCtxt)
            dct_weights[key] = enc_weights[key] + dct_weights[key]
        #print('unencrypted f ',i, HE.decryptFrac(dct_weights['c_0_0'][0][0][0][0] ))
    
    for key in dct_weights:
        dct_weights[key]= dct_weights[key]*denom #c_denom
        #print(dct_weights[key].size())
    
    end = time.time()
    print('Time to aggregate:',end-start)      
    return dct_weights
