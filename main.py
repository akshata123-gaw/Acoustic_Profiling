from flask import Flask, request, jsonify
import os


app=Flask(__name__)

@app.route("/", methods=['POST', 'GET'])
def index():
    if request.method == "POST":
        save_path = os.path.join("", "temp.wav")
        request.files['audio'].save(save_path)
        result = classify("temp.wav")[0]
        print(result)
        os.remove("temp.wav")
        return jsonify(machine = result)

    return "HELLO"



if __name__ == "__main__":
    from logic import *
    app.run(debug=False)

