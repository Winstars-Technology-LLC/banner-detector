import sys
import time

from werkzeug.utils import secure_filename

from test_main import run_testing

import os
sys.path.append('models/')
sys.path.append('models/nn_models/')
print(sys.path)
import yaml
import os
from core.config import app
from core.tools import wrap_response

from models.execution import process_video, Compute
from flask import request, Response, stream_with_context

@app.before_request
def before_request():

    with open(app.config["CONFIG_PATH"], 'r') as file:
        model_parameters = yaml.load(file, Loader=yaml.FullLoader)

    model_parameters["mask_path"] = app.config["MASK_PATH"]
    model_parameters["model_weights_path"] = app.config["WEIGHT_FOLDER"] + '/mrcnn.h5'

    with open(app.config["CONFIG_PATH"], 'w') as write_file:
        documents = yaml.dump(model_parameters, write_file)


@app.route('/')
def init():
    return "<p> Hello World </p>"

@app.route('/periods', methods=["POST"])
def set_time_periods():
    data = request.json

    if not data:
        return wrap_response({}, True, 400)

    with open(app.config["CONFIG_PATH"], 'r') as file:
        model_parameters = yaml.load(file, Loader=yaml.FullLoader)

    model_parameters['periods'] = {}
    if 'periods' in data and data['periods']:
        for period in data['periods']:
            model_parameters['periods'][period] = {}
            start, finish = data['periods'][period].values()
            model_parameters['periods'][period]['start'] = start
            model_parameters['periods'][period]['finish'] = finish

    with open(app.config["CONFIG_PATH"], 'w') as write_file:
        documents = yaml.dump(model_parameters, write_file)

    return wrap_response(data, False, 200)


@app.route('/banner', methods=['POST'])
def select_logo():
    files = request.files

    with open(app.config["CONFIG_PATH"], 'r') as params_file:
        model_parameters = yaml.load(params_file, Loader=yaml.FullLoader)

    banner_names = {"gazprom": 1, "heineken": 2, "mastercard": 3, "nissan": 4, "pepsi": 5, "playstation": 6}

    model_parameters['replace'] = {}
    for name in files:
        logotype = files[name]
        filename = secure_filename(logotype.filename)
        logotype_path = os.path.join(app.config['LOGO_FOLDER'], filename)
        logotype.save(logotype_path)
        model_parameters['replace'][banner_names[name]] = logotype_path

    with open(app.config["CONFIG_PATH"], 'w') as write_file:
        documents = yaml.dump(model_parameters, write_file)

    return wrap_response({}, True, 200)


@app.route('/set_video', methods=["POST"])
def get_video_path():
    video_path = request.form['video_path']

    with open(app.config["CONFIG_PATH"], 'r') as params_file:
        model_parameters = yaml.load(params_file, Loader=yaml.FullLoader)

    model_parameters['source_link'] = video_path

    video_name = video_path.split('/')[-1]

    model_parameters['saving_link'] = os.path.join(app.config["DOWNLOAD_FOLDER"], video_name)
    with open(app.config["CONFIG_PATH"], 'w') as write_file:
        documents = yaml.dump(model_parameters, write_file)

    return wrap_response({}, False, 200)


@app.route('/process', methods=["POST"])
def process_video():

    thread_a = Compute(request.__copy__)
    thread_a.run()

    return wrap_response({"Processing in background": True}, False, 200)


if __name__ == '__main__':
    print('Testing')
    # run_testing()
    print('Successful test!!!')

    app.run(host="0.0.0.0", port="5089")
