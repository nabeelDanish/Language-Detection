# import all the necessary libraries
import scipy
from scipy.io import wavfile
from scipy import signal
import numpy as np
import matplotlib.pyplot as plt
from pydub import AudioSegment
from pydub.utils import make_chunks
from pydub.silence import split_on_silence
import random
import librosa
import librosa.display
import os, shutil
import glob
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Activation, BatchNormalization, Dropout
from keras.models import Sequential
from keras.callbacks import TensorBoard, ModelCheckpoint
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv1D, Input, MaxPooling1D, LSTM
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import backend as K
from keras import optimizers, losses, activations, models
import statistics 
from statistics import mode
import time
K.clear_session()

# Defining Global Variables
labels = ['arabic', 'english', 'pashto', 'sindhi', 'urdu']

def getModel():
    db = 0.3

    inputs = Input(shape=(128,32))

    #First Conv1D layer
    conv = Conv1D(32,6, padding='valid', activation='relu', strides=1)(inputs)
    conv = MaxPooling1D(2)(conv)
    conv = Dropout(db)(conv)

    #Second Conv1D layer
    conv = Conv1D(64, 4, padding='valid', activation='relu', strides=1)(conv)
    # conv = MaxPooling1D(2)(conv)
    conv = Dropout(db)(conv)

    conv=LSTM(units=64, return_sequences=True)(conv)
    conv=LSTM(units=128, return_sequences=True)(conv)

    #Flatten layer
    conv = Flatten()(conv)

    #Dense Layer 1
    # conv = Dense(256, activation='relu')(conv)
    # conv = Dropout(0.3)(conv)

    # Dense Layer 2
    conv = Dense(128, activation='relu')(conv)

    outputs = Dense(5, activation='softmax')(conv)

    model_2 = Model(inputs, outputs)
    return model_2

# Function to load the model weights and archtitecture
def initModel(model, weights):
  loaded_model = getModel()

  # load weights into new model
  loaded_model.load_weights(weights)
  return loaded_model

# Function to create a spectogram from a given file and save it in the desired
# destination
def clearFolder(folder):
  for filename in os.listdir(folder):
    file_path = os.path.join(folder, filename)
    try:
      if os.path.isfile(file_path) or os.path.islink(file_path):
        os.unlink(file_path)
      elif os.path.isdir(file_path):
        shutil.rmtree(file_path)
    except Exception as e:
      print('Failed to delete %s. Reason: %s' % (file_path, e))

# Function to split audio file into smaller samples
def splitAudio(input_file, extension, chunk_size, output_path):

  myaudio = AudioSegment.from_file(input_file , extension) 
  myaudio = myaudio.low_pass_filter(3000).high_pass_filter(200)
  chunk_length_ms = chunk_size * 1000 
  chunks = make_chunks(myaudio, chunk_length_ms) 

  #Export all of the individual chunks as wav files
  for i, chunk in enumerate(chunks):
    chunk_name = output_path + "chunk{0}.wav".format(i)
    chunk.export(chunk_name, format="wav")

# Function to process audio by splitting it and converting
# samples to MFCCs
def processAudio(input_file, extension, chunk_file):

  # Clearing Folders
  clearFolder(chunk_file)

  # Splitting the audio
  output_path = chunk_file
  splitAudio(input_file, extension, 1, output_path)

  input_paths = glob.glob(output_path + '*')

  X_pred = []
  data = input_paths
  for i, d in zip(range(len(data)), data):
    SR = 16000
    y, sr = librosa.load(d, sr = SR)
    S = librosa.feature.melspectrogram(y, sr)
    S = librosa.power_to_db(S, ref=np.max)
    if (S.shape == (128, 32)):
      X_pred.append(S)
  return np.array(X_pred)

def getPredictions(inputFile, extension, model, chunk_file):

  # Processing the Audio file to get the final spectograms
  X_pred = processAudio(inputFile, extension, chunk_file)
  # Making Prediction
  pred = model.predict(X_pred)

  # Formating Outputs
  predicted_class_indices=np.argmax(pred, axis=1)

  # Labels
  predictions = [labels[k] for k in predicted_class_indices]

  return predictions

def languageDetect(inputFile, extension, chunk_file):
  model = initModel('model.json', 'weights.hdf5')
  model.compile(loss = 'categorical_crossentropy', optimizer = 'adamax', metrics = ['accuracy'])
  pred = getPredictions(inputFile, extension, model, chunk_file)
  return mode(pred)