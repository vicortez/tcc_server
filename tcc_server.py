import os
import json
from flask import Flask, flash, request
from flask_restful import Resource, Api, reqparse
from subprocess import check_output
import numpy as np
import cv2
import classifiers.custom_classifier_tcc as classification


GLOBAL_COUNT_IN=0
GLOBAL_COUNT_IN_CAR=0
GLOBAL_COUNT_IN_MOT=0
GLOBAL_COUNT_OUT=0
GLOBAL_COUNT_OUT_CAR=0
GLOBAL_COUNT_OUT_MOT=0

class ObjectDetector:

    def __init__(self, object_class=None, confidence=0.2):
        self.objclass = [object_class]  # object of interest
        self.confidence = confidence  # minimum probability

    def countObject(self, frame):
        # grab the frame dimensions and convert it to a blob
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)),
                                     0.007843, (300, 300), 127.5)

        # pass the blob through the network and obtain the detections and
        # predictions

        count_object = 0
        # loop over the detections
        for i in np.arange(0, detections.shape[2]):
            # extract the confidence (i.e., probability) associated with
            # the prediction
            confidence = detections[0, 0, i, 2]

            # filter out weak detections by ensuring the `confidence` is
            # greater than the minimum confidence
            if confidence > self.confidence:
                # extract the index of the class label from the
                # `detections`
                idx = int(detections[0, 0, i, 1])

                # if the predicted class label is not in the set of classes
                # we want then skip the detection
                if CLASSES[idx] not in self.objclass:
                    continue

                count_object += 1
                # compute the (x, y)-coordinates of the bounding box for
                # the object
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")

                # draw the prediction on the frame
                label = "{}: {:.2f}%".format(CLASSES[idx], confidence * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY),
                              COLORS[idx], 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[idx], 2)

        return count_object, frame


# RESTful API classes and functions =======================================
UPLOAD_FOLDER = './images'
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
parser = reqparse.RequestParser()

# Look only in the POST body
# From the request headers
parser.add_argument('Api-Key', location='headers',
                    required=True, choices='dead-beef-4-feed',
                    help='Incorrect. Please, send a valid API-Key.')


class Device(Resource):
    def get(self):
        ipaddr = check_output(['hostname', '-I']).split()
        with open('/proc/uptime', 'r') as f:
            uptime_seconds = float(f.readline().split()[0])
        with open('/proc/loadavg', 'r') as f:
            loadavg = f.readline().split()[0:3]
        for i in range(3):
            loadavg[i] = float(loadavg[i])
        return { 'data': [ { 'device-name': 'Raspberry Pi 3' },
                           { 'device-class': 3 },
                           { 'ip-address' : ipaddr },
                           { 'system-uptime-in-seconds': uptime_seconds },
                           { 'system-load-average': loadavg }
                         ]
               }


class Upload(Resource):
    def post(self):
        global GLOBAL_COUNT_IN
        global GLOBAL_COUNT_IN_CAR
        global GLOBAL_COUNT_IN_MOT
        global GLOBAL_COUNT_OUT
        global GLOBAL_COUNT_OUT_CAR
        global GLOBAL_COUNT_OUT_MOT

        parser.parse_args()
        # check if the post request has the file part
        if ('file' not in request.files) or ('json' not in request.form):
            flash('No file or json part')
            return {'message': 'Error: no file or json part sent'}
        file = request.files['file']
        if file.filename == '':
            flash('No selected file')
            return {'message': 'Error: no file name informed'}
        if file:
            filename = file.filename
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        mdata = json.loads(request.form['json'])
        sensor_id = mdata['sensor_id']
        time_stamp = mdata['time_stamp']
        direction = mdata['direction']
        print("Image received:\nsensor_id = {}\ntime_stamp = {}"
              "\ndirection = {}".format(sensor_id, time_stamp,direction))
        vehicle_type = classification.classify(cv2.imread("images/"+filename))
        if direction == 'l-r':
            GLOBAL_COUNT_IN+=1
            if vehicle_type == 'motorcycle':
                GLOBAL_COUNT_IN_MOT+=1
            else:
                GLOBAL_COUNT_IN_CAR+=1
        elif direction == 'r-l':
            GLOBAL_COUNT_OUT+=1
            if vehicle_type == 'motorcycle':
                GLOBAL_COUNT_OUT_MOT+=1
            else:
                GLOBAL_COUNT_OUT_CAR+=1
        print(str(GLOBAL_COUNT_IN), "veículos entraram no local:"+str(GLOBAL_COUNT_IN_CAR),"carros e",str(GLOBAL_COUNT_IN_MOT),'motocicletas')
        print(str(GLOBAL_COUNT_OUT), "veículos sairam do local:"+str(GLOBAL_COUNT_OUT_CAR),"carros e",str(GLOBAL_COUNT_OUT_MOT),'motocicletas')
        return {"message": "Ok. File received."}


api = Api(app)
api.add_resource(Device, '/device')
api.add_resource(Upload, '/upload')


if __name__ == '__main__':

    app.run(host='0.0.0.0', port='8000')
