from flask import Flask, request, jsonify
import wave
import soundfile as sf
import threading
from helper import getAudio, denoiseAudio, classify
import os
import datetime
import librosa


def rawStore( file_name, folder_name, audio, sr):
    path = os.path.join(folder_name, str(datetime.date.today()))
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(os.path.join(path, file_name), audio, sr)


def denoiseStore(file_name, folder_name, audio, sr):
    path = os.path.join(folder_name, str(datetime.date.today()))
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(os.path.join(path, file_name), audio, sr)


def resultStore(file_name, folder_name, audio, sr, labels):
    path = os.path.join(folder_name, str(datetime.date.today()), labels)
    if not os.path.exists(path):
        os.makedirs(path)
    sf.write(os.path.join(path, file_name), audio, sr)


class ZiApplication(Flask):

    def __init__(self):
        super().__init__(__name__)

        @self.route('/')
        def home():
            return "Welcome to Zi"

        @self.route('/upload', methods=['POST'])
        def upload_audio():
            file_name = request.files['audio_file'].filename
            audio, sr = getAudio(request.files['audio_file'])
            deNoisedaudio, sr = denoiseAudio(audio, sr)
            labels = classify(deNoisedaudio, sr)
            # rawStore(file_name, "raw", audio, sr)
            # denoiseStore(file_name, "Denoise", deNoisedaudio, sr)
            # resultStore(file_name, "Results", audio, sr, labels)
            threading.Thread(target=rawStore, args=[file_name, "raw", audio, sr]).start()
            threading.Thread(target=denoiseStore, args=[file_name, "Denoise", deNoisedaudio, sr]).start()
            threading.Thread(target=resultStore, args=[file_name, "Results", audio, sr, labels]).start()
            return jsonify("File Uploaded")

        @self.route('/endresult', methods=['GET'])
        def endResult():
            response_dic = {}
            currDate = request.get_json()['date']
            format_string = "%Y-%m-%d"
            date = datetime.datetime.strptime(currDate, format_string)
            # Raw Path Count
            raw_path = os.path.join('raw', str(date.date()))
            raw_count = len(os.listdir(raw_path))
            response_dic['total_files'] = raw_count
            # Fault
            fault_list = ['bear_fault', 'gear_fault', 'normal_machine']
            for i in fault_list:
                fault_path = os.path.join('Results', str(date.date()), i)
                if os.path.exists(fault_path):
                    response_dic[i] = len(os.listdir(fault_path))
                else:
                    response_dic[i] = 0
            return jsonify(response_dic)



if __name__ == '__main__':
    app = ZiApplication()
    app.run(debug=True)
