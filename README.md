
# Language Detection on Audio Files
Nabeel Danish

A spectogram-based approach to identifying langauges most commonly spoken in Pakistan.
this includes Urdu, English, Arabic, Pashto, and Sindhi. The model is contructed as CNN with LSTM layers
achieving an accuracy of 95%

# Usage

import the file languageDetection.py to use in your python script

# Functions
```
def languageDetect(inputFile, extension, chunk_file)
```
	Parameters:
	
	inputFile -- path to file for audio
	extension -- audio file extension
	chunk_file -- path to folder where the model stores preprocessing data

	Return Value:

	pred -- string literal indicating the language most frequent found in audio

# Dependancies
Tensorflow
Keras
Numpy
Scipy
Pydub
Librosa
