import numpy as np
from flask import Flask, request, jsonify, send_file, send_from_directory
from markupsafe import escape

from utils import is_valid_access_code

app = Flask(__name__)


@app.route('/get_model', methods=["POST"])
def get_model():
    if is_valid_access_code(request.json["access_code"]):
        model = "models/weights_coscia.hdf5"
        return send_file(model), 200
    else:
        return {"message": "invalid access code"}, 401


@app.route('/upload_model', methods=['POST'])
def upload_file():
    f = request.files['upload_file']
    f.save('models/uploaded_file.hdf5')
    content = request.json
    print(content)
    return {"message": "upload successfull"}, 200


if __name__ == '__main__':
    app.run(debug=True)
