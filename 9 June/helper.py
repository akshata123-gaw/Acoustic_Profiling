import wave
import tensorflow as tf
import librosa
import os
import numpy as np
model = tf.keras.models.load_model(os.path.join("models", "model.h5"))

classes = ['bear_fault', 'gear_fault',  'normal_machine']

def getAudio(file):
    file.save("tmp.wav")
    audio, sr = librosa.load(os.path.join("tmp.wav"))
    os.remove(os.path.join("tmp.wav"))
    return audio, sr

def denoiseAudio(y, sr):
    S_full = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=1024)
    S_filter = librosa.decompose.nn_filter(S_full,
                                       aggregate=np.median,
                                       metric='cosine',
                                       width=int(librosa.time_to_frames(1, sr=sr)))

    S_filter = np.minimum(S_full, S_filter)
    S = librosa.feature.inverse.mel_to_stft(S_filter)
    y = librosa.griffinlim(S)
    return y, sr



def classify(audio, sample_rate):

    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
    mfccs_scaled_features = np.mean(mfccs_features.T,axis=0)
    mfccs_scaled_features=mfccs_scaled_features.reshape(1,-1)
    predicted_label=np.argmax(model.predict(mfccs_scaled_features),axis=-1)[0]

    return classes[predicted_label]