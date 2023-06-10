import wave
import soundfile as sf
import threading
from helper import getAudio, denoiseAudio, classify
import os
import datetime
import librosa
from flask import Flask


class ZiMachine:

    def __init__(self):
        self.app = Flask(__name__)
        self.path = os.path.join('raw', str(datetime.date.today()))
        self.thread = threading.Thread(target=self.processing, args=[15, 11], daemon=True)

    def home(self):
        return "Welcome to Zi"

    def upload_audio(self):
        audio_file, sr = getAudio(request.files['audio_file'])
        filename = request.files['audio_file'].filename
        if not os.path.exists(self.path):
            os.makedirs(self.path)
        sf.write(os.path.join(self.path, filename), audio_file, sr)
        return jsonify("File Uploaded")

    def endResult(self):
        response_dic = {}
        currDate = request.get_json()['date']
        print(currDate)
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
            print(fault_path, i)
            if os.path.exists(fault_path):
                response_dic[i] = len(os.listdir(fault_path))
            else:
                response_dic[i] = 0 
        return jsonify(response_dic)

    def processing(self, hr, min):
        print("I am called")
        while 1:
            if datetime.datetime.now().hour==hr and datetime.datetime.now().minute==min:
                listdir  = os.listdir(self.path)
                for file_name in listdir:
                    if  file_name.endswith('.wav'):
                        print(file_name)
                        audio, sample_rate = librosa.load(os.path.join(self.path, file_name))
                        deNoisedaudio, sample_rate = denoiseAudio(audio, sample_rate)
                        labels = classify(deNoisedaudio, sample_rate)
                        # Denoised
                        path_den = os.path.join('Denoise', str(datetime.date.today()))
                        if not os.path.exists(path_den):
                            os.makedirs(path_den)
                        sf.write(os.path.join(path_den, file_name), deNoisedaudio, sample_rate)
                        # Fault 
                        path_fl = os.path.join('Results', str(datetime.date.today()), labels)
                        if not os.path.exists(path_fl):
                            os.makedirs(path_fl)
                        sf.write(os.path.join(path_fl, file_name), audio, sample_rate)
                        print(file_name, "is processed")
            return None
        
    def run(self):
        self.thread.start()
        self.app.run()


if __name__ == '__main__':
    zi_machine = ZiMachine()
    zi_machine.run()
