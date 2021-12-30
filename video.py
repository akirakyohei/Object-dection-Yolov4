#! /usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division, print_function, absolute_import
import sys
#sys.path.remove('/opt/ros/kinetic/lib/python2.7/dist-packages')
import os
import datetime
from timeit import time
import warnings
import cv2
import numpy as np
import argparse
from PIL import Image
<<<<<<< HEAD
from yolov4 import YOLOV4
=======
from yolo import YOLO
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
<<<<<<< HEAD
from deep_sort.detection_yolo import Detection_YOLO
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from deep_sort import nn_matching as nn_matching
from collections import deque
from tensorflow.python.keras import backend as K
from collections import Counter
import tensorflow as tf
from yolo4.utils import resize_image, load_class_names
from config import width, height
import datetime
import math
=======
from deep_sort.tracker import Tracker
from tools import generate_detections as gdet
from deep_sort.detection import Detection as ddet
from collections import deque
from tensorflow.python.keras import backend
import tensorflow as tf
from yolo4.utils import resize_image, load_class_names
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
# config = tf.compat.v1.ConfigProto
# config.gpu_options.allow_growth = False
# session = tf.compat.v1.InteractiveSession(config=config)

<<<<<<< HEAD
tf.compat.v1.disable_eager_execution()
=======

>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input",help="path to input video", default = "./test_video/TownCentreXVID.avi")
ap.add_argument("-c", "--class",help="name of class", default = "person")
args = vars(ap.parse_args())

pts = [deque(maxlen=30) for _ in range(9999)]
warnings.filterwarnings('ignore')

# initialize a list of colors to represent each possible class label
np.random.seed(100)
COLORS = np.random.randint(0, 255, size=(200, 3),
	dtype="uint8")
#list = [[] for _ in range(100)]
<<<<<<< HEAD
memory={}

def main(yolo):
=======

def main(yolo):

>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
    start = time.time()
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0
<<<<<<< HEAD
    show_detections = False
    tracking =True

    counter = []
    #deep_sort
    model_filename = 'model_data/mars-small128.pb'
=======

    counter = []
    #deep_sort
    model_filename = 'model_data/market1501.pb'
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
    encoder = gdet.create_box_encoder(model_filename,batch_size=1)

    find_objects = ['person']
    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    tracker = Tracker(metric)

    writeVideo_flag = True
    video_capture = cv2.VideoCapture(args["input"])
    #video_capture = cv2.VideoCapture(0)

<<<<<<< HEAD
    count_dict ={}
=======
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
    if writeVideo_flag:
    # Define the codec and create VideoWriter object
        w = int(video_capture.get(3))
        h = int(video_capture.get(4))
        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        out = cv2.VideoWriter('./output/output.avi',fourcc, 15, (w,h))
        list_file = open('detection_rslt.txt', 'w')
        frame_index = -1

    fps = 0.0

    while True:

        ret, frame = video_capture.read()  # frame shape 640*480*3
        if ret != True:
            break
<<<<<<< HEAD
        if frame is None:
            break
=======
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
        t1 = time.time()

        #image = Image.fromarray(frame)
        image = Image.fromarray(frame[...,::-1]) #bgr to rgb
        resized_frame = tf.expand_dims(frame, 0)
<<<<<<< HEAD
        resized_frame = resize_image(
        resized_frame, (width,height))

        boxes, confidence, classes = yolo.detect_image(image)
        features = encoder(frame,boxes)
        # score to 1.0 here).
        if tracking: 
            detections = [Detection(bbox, confidence, cls, feature) for bbox, confidence, cls, feature in
                          zip(boxes, confidence, classes, features)]
        else:
            detections = [Detection_YOLO(bbox, confidence, cls) for bbox, confidence, cls in
                          zip(boxes, confidence, classes)]

        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        # classes = np.array([d.cls for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        class_counter=Counter()

        # Call the tracker
        if tracking:
            tracker.predict()
            tracker.update(detections)
            track_count =int(0)
            for track in tracker.tracks:
                if not track.is_confirmed() or track.time_since_update > 1:
                        continue
                bbox = track.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 255),
                                  1)  # WHITE BOX
                cv2.putText(frame, "ID: " + str(track.track_id), (int(bbox[0]), int(bbox[1])), 0,
                                1.5e-3 * frame.shape[0], (0, 255, 0), 1)

                track_count += 1  # add 1 for each tracked object

            cv2.putText(frame, "Current total count: " + str(track_count), (int(20), int(60 * 5e-3 * frame.shape[0])), 0, 2e-3 * frame.shape[0],
                            (255, 255, 255), 2)  

        det_count = int(0)
        for det in detections:
            bbox = det.to_tlbr()
            score = "%.2f" % (det.confidence * 100) + "%"
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 0, 0),
                              1)  # BLUE BOX
            if len(classes) > 0:
                cls = det.cls
                cv2.putText(frame, str(cls) + " " + score, (int(bbox[0]), int(bbox[3])), 0,
                                1.5e-3 * frame.shape[0], (0, 255, 0), 1)
                class_counter[cls] += 1
            det_count += 1

            # display counts for each class as they appear
        y = 80 * 5e-3 * frame.shape[0]
        for cls in class_counter:
            class_count = class_counter[cls]
            cv2.putText(frame, str(cls) + " " + str(class_count), (int(20), int(y)), 0, 2e-3 * frame.shape[0],
                            (255, 255, 255), 2)
            y += 20 * 5e-3 * frame.shape[0] #TODO apply this to other text

            # use YOLO counts if tracking is turned off
        if tracking:
            count = track_count
        else:
            count = det_count
=======
        resized_frame = resize_image(resized_frame, (608,608))

        boxs, confidence, class_names = yolo.detect_image(image,args["class"])
        features = encoder(frame,boxs)
        # score to 1.0 here).
        detections = [Detection(bbox, 1.0, feature) for bbox, feature in zip(boxs, features)]
        # Run non-maxima suppression.
        boxes = np.array([d.tlwh for d in detections])
        scores = np.array([d.confidence for d in detections])
        indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
        detections = [detections[i] for i in indices]

        # Call the tracker
        tracker.predict()
        tracker.update(detections)

        i = int(0)
        indexIDs = []
        c = []
        boxes = []

        for det in detections:
            bbox = det.to_tlbr()
            cv2.rectangle(frame,(int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(255,255,255), 2)
            #print(class_names)
            #print(class_names[p])

        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            #boxes.append([track[0], track[1], track[2], track[3]])
            indexIDs.append(int(track.track_id))
            counter.append(int(track.track_id))
            bbox = track.to_tlbr()
            color = [int(c) for c in COLORS[indexIDs[i] % len(COLORS)]]
            #print(frame_index)
            list_file.write(str(frame_index)+',')
            list_file.write(str(track.track_id)+',')
            cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])),(color), 3)
            b0 = str(bbox[0])#.split('.')[0] + '.' + str(bbox[0]).split('.')[0][:1]
            b1 = str(bbox[1])#.split('.')[0] + '.' + str(bbox[1]).split('.')[0][:1]
            b2 = str(bbox[2]-bbox[0])#.split('.')[0] + '.' + str(bbox[3]).split('.')[0][:1]
            b3 = str(bbox[3]-bbox[1])

            list_file.write(str(b0) + ','+str(b1) + ','+str(b2) + ','+str(b3))
            #print(str(track.track_id))
            list_file.write('\n')
            #list_file.write(str(track.track_id)+',')
            cv2.putText(frame,str(track.track_id),(int(bbox[0]), int(bbox[1] -50)),0, 5e-3 * 150, (color),2)
            if len(class_names) > 0:
               class_name = class_names[0]
               cv2.putText(frame, str(class_names[0]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (color),2)

            i += 1
            #bbox_center_point(x,y)
            center = (int(((bbox[0])+(bbox[2]))/2),int(((bbox[1])+(bbox[3]))/2))
            #track_id[center]

            pts[track.track_id].append(center)

            thickness = 5
            #center point
            cv2.circle(frame,  (center), 1, color, thickness)

			# draw motion path
            for j in range(1, len(pts[track.track_id])):
                if pts[track.track_id][j - 1] is None or pts[track.track_id][j] is None:
                   continue
                thickness = int(np.sqrt(64 / float(j + 1)) * 2)
                cv2.line(frame,(pts[track.track_id][j-1]), (pts[track.track_id][j]),(color),thickness)
                #cv2.putText(frame, str(class_names[j]),(int(bbox[0]), int(bbox[1] -20)),0, 5e-3 * 150, (255,255,255),2)

        count = len(set(counter))
        cv2.putText(frame, "Total Pedestrian Counter: "+str(count),(int(20), int(120)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "Current Pedestrian Counter: "+str(i),(int(20), int(80)),0, 5e-3 * 200, (0,255,0),2)
        cv2.putText(frame, "FPS: %f"%(fps),(int(20), int(40)),0, 5e-3 * 200, (0,255,0),3)
        cv2.namedWindow("YOLO4_Deep_SORT", 0);
        cv2.resizeWindow('YOLO4_Deep_SORT', 1024, 768);
        cv2.imshow('YOLO4_Deep_SORT', frame)

>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4

        if writeVideo_flag:
            # save a frame
            out.write(frame)
            frame_index = frame_index + 1


        fps  = ( fps + (1./(time.time()-t1)) ) / 2
        out.write(frame)
        frame_index = frame_index + 1

        # Press Q to stop!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    print(" ")
    print("[Finish]")
    end = time.time()

<<<<<<< HEAD
    # if len(pts[track.track_id]) != None:
    #    print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    # else:
    #    print("[No Found]")
=======
    if len(pts[track.track_id]) != None:
       print(args["input"][43:57]+": "+ str(count) + " " + str(class_name) +' Found')

    else:
       print("[No Found]")
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
	#print("[INFO]: model_image_size = (960, 960)")
    video_capture.release()
    if writeVideo_flag:
        out.release()
        list_file.close()
    cv2.destroyAllWindows()

if __name__ == '__main__':
<<<<<<< HEAD
    main(YOLOV4())
=======
    main(YOLO())
>>>>>>> 55b161656bff0bbf0eb71ec75f5320cba7b97da4
