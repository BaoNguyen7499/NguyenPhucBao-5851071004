
import numpy as np
import math
from tkinter import *
from tkinter.ttk import *
import tkinter as tk
from tkinter import ttk
# from tkinter import ttk
# from time import sleep
# from threading import Thread
from tkinter import filedialog
# import tkinter
import cv2
from PIL import Image
from PIL import ImageTk
import random
# import argparse
import imutils
import time


START_POINT = 80
END_POINT = 150

START_POINT_HP1 = 550
END_POINT_HP1 = 400
START_POINT_HP2 = 400
END_POINT_HP2 = 150

START_POINT_W = 80
END_POINT_W = 550

START_POINT_H = 440
END_POINT_H = 200

START_POINT_L = 640
END_POINT_L = 440
LANE_ERROR = 600
CLASSES = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck",
           "boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
           "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe",
           "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
           "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
           "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana",
           "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake",
           "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse",
           "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
           "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush"]
# Define vehicle class
VEHICLE_CLASSES = [1, 2, 3, 5, 6, 7]

YOLOV4_CFG = 'cfg/yolov4.cfg'
YOLOV4_WEIGHT = 'weights/yolov4.weights'

CONFIDENCE_SETTING = 0.4
YOLOV4_WIDTH = 320
YOLOV4_HEIGHT = 320

MAX_DISTANCE = 80
a = None
start_time = time.time()
frame_id = 0
fps = 0
speed = [None] * 1000

pixels_per_meter = 1
Tracker = {}
StartPosition = {}
CurrentPosition = {}


def get_output_layers(net):
    """
    Get output layers of darknet
    :param net: Model
    :return: output_layers
    """

    try:
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        return output_layers
    except:
        print("Can't get output layers")
        return None


def detections_yolo4(net, image, confidence_setting, yolo_w, yolo_h, frame_w, frame_h, classes=None):
    """
    Detect object use yolo3 model
    :param net: model
    :param image: image
    :param confidence_setting: confidence setting
    :param yolo_w: dimension of yolo input
    :param yolo_h: dimension of yolo input
    :param frame_w: actual dimension of frame
    :param frame_h: actual dimension of frame
    :param classes: name of object
    :return:
    """
    img = cv2.resize(image, (yolo_w, yolo_h))
    blob = cv2.dnn.blobFromImage(img, 0.00392, (yolo_w, yolo_h), swapRB=True, crop=False)
    net.setInput(blob)
    layer_output = net.forward(get_output_layers(net))

    boxes = []
    class_ids = []
    confidences = []

    for out in layer_output:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > confidence_setting and class_id in VEHICLE_CLASSES:
                print("Object name: " + classes[class_id] + " - Confidence: {:0.2f}".format(confidence * 100))
                center_x = int(detection[0] * frame_w)
                center_y = int(detection[1] * frame_h)
                w = int(detection[2] * frame_w)
                h = int(detection[3] * frame_h)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
    return boxes, class_ids, confidences

def draw_prediction(classes, colors, img, class_id, confidence, x, y, width, height, speed):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """

    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if speed > 30:
            cv2.putText(img, str(round(speed)) + " km/h",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
        if check_start_lane(x, width):
            if class_id == 3 or class_id == 1:
                cv2.putText(img, "Lane encroach", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
        if check_start_lane_Car(x, width):
            if class_id == 2 or class_id == 5 or class_id == 7:
                cv2.putText(img, "Lane encroach", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 0, 255), 2)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))
    return img

def draw_prediction_h(classes, colors, img, class_id, confidence, x, y, width, height, speed):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """

    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        if speed > 10:
            cv2.putText(img, str(round(speed)) + " km/h",
                        (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (0, 255, 255), 1)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))
    return img

def draw_prediction1(classes, colors, img, class_id, confidence, x, y, width, height):
    """
    Draw bounding box and put classe text and confidence
    :param classes: name of object
    :param colors: color for object
    :param img: immage
    :param class_id: class_id of this object
    :param confidence: confidence
    :param x: top, left
    :param y: top, left
    :param width: width of bounding box
    :param height: height of bounding box
    :return: None
    """

    try:
        label = str(classes[class_id])
        color = colors[class_id]
        center_x = int(x + width / 2.0)
        center_y = int(y + height / 2.0)
        x = int(x)
        y = int(y)
        width = int(width)
        height = int(height)

        cv2.rectangle(img, (x, y), (x + width, y + height), color, 1)
        cv2.circle(img, (center_x, center_y), 2, (0, 255, 0), -1)
        cv2.putText(img, label + ": {:0.2f}%".format(confidence * 100), (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        cv2.putText(img, "{:0.0f}".format(random.randint(60, 80)) + " km/h",
                    (x, y - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1)
        # if check_start_lane(x, width):
        #     if class_id == 3 or class_id == 1:
        #         cv2.putText(img, "error", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                     (0, 0, 255), 2)
        # if check_start_lane_Car(x, width):
        #     if class_id == 2 or class_id == 5 or class_id == 7:
        #         cv2.putText(img, "error", (x, y-20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #                     (0, 0, 255), 2)
    except (Exception, cv2.error) as e:
        print("Can't draw prediction for class_id {}: {}".format(class_id, e))

def check_location(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT:
        return True
    else:
        return False


def check_start_line(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > START_POINT:
        return True
    else:
        return False


def check_start_lane(box_x, box_width):
    """
    Check center point of object that passing start lane or not
    :param box_x: y value of bounding box
    :param box_width: height of bounding box
    :return: Boolean
    """
    center_y = int(box_x + box_width / 2.0)
    if center_y > LANE_ERROR:
        return True
    else:
        return False

def check_start_line2(box_x, box_width):
    """
    Check center point of object that passing start lane or not
    :param box_x: y value of bounding box
    :param box_width: height of bounding box
    :return: Boolean
    """
    center_y = int(box_x + box_width / 2.0)
    if center_y > START_POINT_H:
        return True
    else:
        return False

def check_start_lane_Car(box_x, box_width):
    """
    Check center point of object that passing start lane or not
    :param box_x: y value of bounding box
    :param box_width: height of bounding box
    :return: Boolean
    """
    center_y = int(box_x + box_width / 2.0)
    if center_y < LANE_ERROR:
        return True
    else:
        return False
def calculate_speed(startPosition, currentPosition, fps):

    global pixels_per_meter

    # speed calculation pixel
    distance_in_pixels = math.sqrt(math.pow(currentPosition[0] - startPosition[0], 2)
                                   + math.pow(currentPosition[1] - startPosition[1], 2))

    # speed calculation by meter
    distance_in_meters = distance_in_pixels / pixels_per_meter

    # speed calculation by m/s
    speed_in_meter_per_second = distance_in_meters * fps
    # swap km/h
    speed_in_kilometer_per_hour = speed_in_meter_per_second * 3.6

    return speed_in_kilometer_per_hour

def counting_vehicle(skip_frame=1):
    global file_Name, cap
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG, YOLOV4_WEIGHT)

    # Read first frame
    # cap = cv2.VideoCapture(file_Name)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('vehicles.avi', video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    count_car = 0
    count_bi = 0
    count_motorbike = 0
    count_bus = 0
    count_train = 0
    count_truck = 0
    number_id = 0
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height, speed[idx])
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    # This object passed the end line
                    number_vehicle += 1
                    if class_id == 1:
                        count_bi += 1
                    if class_id == 2:
                        count_car += 1
                    if class_id == 3:
                        count_motorbike += 1
                    if class_id == 5:
                        count_bus += 1
                    if class_id == 6:
                        count_train += 1
                    if class_id == 7:
                        count_truck += 1
                    cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (0, 0, 255), 3)
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo4(net, frame, CONFIDENCE_SETTING, YOLOV4_WIDTH,
                                                             YOLOV4_HEIGHT, width, height, classes=CLASSES)
            # for ID in Tracker.keys:
            #     trackedPosition = Tracker[ID].get_position()
            #     t_x = int(trackedPosition.left())
            #     t_y = int(trackedPosition.top())
            #     t_w = int(trackedPosition.width())
            #     t_h = int(trackedPosition.height())
            #     # Tinh tam diem cua car da track
            #     t_x_center = t_x + 0.5 * t_w
            #     t_y_center = t_y + 0.5 * t_h
            #
            #     # Kiem tra xem co phai ca da track hay khong
            #     # if (t_x <= x_center <= (t_x + t_w)) and (t_y <= y_center <= (t_y + t_h)) and (
            #     #         x <= t_x_center <= (x + w)) and (y <= t_y_center <= (y + h)):
            #     #     matchCarID = ID
            #     Tracker[number_id] = tracker
            #     StartPosition[number_id] = [t_x, t_y, t_w, t_h]
            # 
            #     number_id += 1
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    StartPosition[idx] = box_x, box_y, box_width, box_height
                    number_id += 1
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)

                        CurrentPosition[idx] = [current_box_x, current_box_y, current_box_width,
                                                current_box_height]
                        # endtime = time.time() - start_time
                        # fps = number_frame / endtime
                        endtime = time.time()
                        fps = 1/(endtime-start_time)
                        # track and speed calculation
                        [x1, y1, w1, h1] = StartPosition[idx]
                        [x2, y2, w2, h2] = CurrentPosition[idx]

                        StartPosition[idx] = [x2, y2, w2, h2]
                        # moving object
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            speed[idx] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height) and check_start_line2(box_x, box_width):
                        # Append new object to list
                        new_tracker = cv2.legacy.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }

                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height, speed[idx])
        #FPS
        endtime = time.time() - start_time
        fps = number_frame / endtime

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Put summary text
        cv2.putText(frame, "Total Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Motorbike Number : {:03d}".format(count_motorbike), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Car Number : {:03d}".format(count_car), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bicycle Number : {:03d}".format(count_bi), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bus Number : {:03d}".format(count_bus), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # cv2.putText(frame, "Train Number : {:03d}".format(count_train), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)
        cv2.putText(frame, "Truck Number : {:03d}".format(count_truck), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw lane
        cv2.line(frame, (START_POINT_L, 0), (END_POINT_L, width), (204, 255, 208), 2)
        # Draw start line
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (255, 0, 0), 2)
        # Show frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
        cv2.imshow("Counting", frame)
        # a = (1400, 768)
        # b = cv2.resize(frame, a, interpolation=cv2.INTER_AREA)
        # cv2.imshow("c" , b)

    cap.release()
    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def counting_vehicle2(skip_frame=1):
    global file_Name, cap
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG, YOLOV4_WEIGHT)

    # Read first frame
    # cap = cv2.VideoCapture(file_Name)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('vehicles.avi', video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    count_car = 0
    count_bi = 0
    count_motorbike = 0
    count_bus = 0
    count_train = 0
    count_truck = 0
    number_id = 0
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction1(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location(box_y, box_height, height):
                    # This object passed the end line
                    number_vehicle += 1
                    if class_id == 1:
                        count_bi += 1
                    if class_id == 2:
                        count_car += 1
                    if class_id == 3:
                        count_motorbike += 1
                    if class_id == 5:
                        count_bus += 1
                    if class_id == 6:
                        count_train += 1
                    if class_id == 7:
                        count_truck += 1
                    cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (0, 0, 255), 3)
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo4(net, frame, CONFIDENCE_SETTING, YOLOV4_WIDTH,
                                                             YOLOV4_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    StartPosition[idx] = box_x, box_y, box_width, box_height
                    number_id += 1
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)

                        CurrentPosition[idx] = [current_box_x, current_box_y, current_box_width,
                                                current_box_height]
                        # endtime = time.time() - start_time
                        # fps = number_frame / endtime
                        endtime = time.time()
                        fps = 1/(endtime-start_time)
                        # track and speed calculation

                        [x1, y1, w1, h1] = StartPosition[idx]
                        [x2, y2, w2, h2] = CurrentPosition[idx]

                        StartPosition[idx] = [x2, y2, w2, h2]
                        # moving object
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            speed[idx] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line(box_y, box_height):
                        # Append new object to list
                        new_tracker = cv2.legacy.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }

                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction1(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        #FPS
        endtime = time.time() - start_time
        fps = number_frame / endtime

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Put summary text
        cv2.putText(frame, "Total Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Motorbike Number : {:03d}".format(count_motorbike), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Car Number : {:03d}".format(count_car), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bicycle Number : {:03d}".format(count_bi), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bus Number : {:03d}".format(count_bus), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # cv2.putText(frame, "Train Number : {:03d}".format(count_train), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)
        cv2.putText(frame, "Truck Number : {:03d}".format(count_truck), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw lane
        # cv2.line(frame, (START_POINT_L, 0), (END_POINT_L, width), (204, 255, 208), 2)
        # Draw start line
        cv2.line(frame, (0, START_POINT), (width, START_POINT), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (0, height - END_POINT), (width, height - END_POINT), (255, 0, 0), 2)
        # Show frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
        cv2.imshow("Counting", frame)
        # return frame
    cap.release()
    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_location2(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > height - END_POINT_HP2:
        return True
    else:
        return False


def check_start_line4(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y > START_POINT_HP2:
        return True
    else:
        return False

def counting_vehicle3(skip_frame=1):
    global file_Name, cap
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG, YOLOV4_WEIGHT)

    # Read first frame
    # cap = cv2.VideoCapture(file_Name)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('vehicles.avi', video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    count_car = 0
    count_bi = 0
    count_motorbike = 0
    count_bus = 0
    count_train = 0
    count_truck = 0
    number_id = 0
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction1(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location2(box_y, box_height, height):
                    # This object passed the end line
                    number_vehicle += 1
                    if class_id == 1:
                        count_bi += 1
                    if class_id == 2:
                        count_car += 1
                    if class_id == 3:
                        count_motorbike += 1
                    if class_id == 5:
                        count_bus += 1
                    if class_id == 6:
                        count_train += 1
                    if class_id == 7:
                        count_truck += 1
                    cv2.line(frame, (650, height - END_POINT_HP2), (width, height - END_POINT_HP2), (0, 0, 255), 3)
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo4(net, frame, CONFIDENCE_SETTING, YOLOV4_WIDTH,
                                                             YOLOV4_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    StartPosition[idx] = box_x, box_y, box_width, box_height
                    number_id += 1
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)

                        CurrentPosition[idx] = [current_box_x, current_box_y, current_box_width,
                                                current_box_height]
                        # endtime = time.time() - start_time
                        # fps = number_frame / endtime
                        endtime = time.time()
                        fps = 1/(endtime-start_time)
                        # track and speed calculation

                        [x1, y1, w1, h1] = StartPosition[idx]
                        [x2, y2, w2, h2] = CurrentPosition[idx]

                        StartPosition[idx] = [x2, y2, w2, h2]
                        # moving object
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            speed[idx] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line4(box_y, box_height):
                        # Append new object to list
                        new_tracker = cv2.legacy.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }

                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction1(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        #FPS
        endtime = time.time() - start_time
        fps = number_frame / endtime

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Put summary text
        cv2.putText(frame, "Total Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Motorbike Number : {:03d}".format(count_motorbike), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Car Number : {:03d}".format(count_car), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bicycle Number : {:03d}".format(count_bi), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bus Number : {:03d}".format(count_bus), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # cv2.putText(frame, "Train Number : {:03d}".format(count_train), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)
        cv2.putText(frame, "Truck Number : {:03d}".format(count_truck), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw lane
        # cv2.line(frame, (START_POINT_L, 0), (END_POINT_L, width), (204, 255, 208), 2)
        # Draw start line
        cv2.line(frame, (650, START_POINT_HP2), (width, START_POINT_HP2), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (650, height - END_POINT_HP2), (width, height - END_POINT_HP2), (255, 0, 0), 2)
        # Show frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
        cv2.imshow("Counting", frame)
        # return frame
    cap.release()
    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_location3(box_y, box_height, height):
    """
    Check center point of object that passing end line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :param height: height of image
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y < height - END_POINT_HP1:
        return True
    else:
        return False


def check_start_line3(box_y, box_height):
    """
    Check center point of object that passing start line or not
    :param box_y: y value of bounding box
    :param box_height: height of bounding box
    :return: Boolean
    """
    center_y = int(box_y + box_height / 2.0)
    if center_y < START_POINT_HP1:
        return True
    else:
        return False

def counting_vehicle4(skip_frame=1):
    global file_Name, cap
    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG, YOLOV4_WEIGHT)

    # Read first frame
    # cap = cv2.VideoCapture(file_Name)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('vehicles.avi', video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    count_car = 0
    count_bi = 0
    count_motorbike = 0
    count_bus = 0
    count_train = 0
    count_truck = 0
    number_id = 0
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction1(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height)
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location3(box_y, box_height, height):
                    # This object passed the end line
                    number_vehicle += 1
                    if class_id == 1:
                        count_bi += 1
                    if class_id == 2:
                        count_car += 1
                    if class_id == 3:
                        count_motorbike += 1
                    if class_id == 5:
                        count_bus += 1
                    if class_id == 6:
                        count_train += 1
                    if class_id == 7:
                        count_truck += 1
                    cv2.line(frame, (0, height - END_POINT_HP1), (600, height - END_POINT_HP1), (0, 0, 255), 3)
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo4(net, frame, CONFIDENCE_SETTING, YOLOV4_WIDTH,
                                                             YOLOV4_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location3(box_y, box_height, height):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    StartPosition[idx] = box_x, box_y, box_width, box_height
                    number_id += 1
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)

                        CurrentPosition[idx] = [current_box_x, current_box_y, current_box_width,
                                                current_box_height]
                        # endtime = time.time() - start_time
                        # fps = number_frame / endtime
                        endtime = time.time()
                        fps = 1/(endtime-start_time)
                        # track and speed calculation

                        [x1, y1, w1, h1] = StartPosition[idx]
                        [x2, y2, w2, h2] = CurrentPosition[idx]

                        StartPosition[idx] = [x2, y2, w2, h2]
                        # moving object
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            speed[idx] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line3(box_y, box_height):
                        # Append new object to list
                        new_tracker = cv2.legacy.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }

                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction1(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height)
        #FPS
        endtime = time.time() - start_time
        fps = number_frame / endtime

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Put summary text
        cv2.putText(frame, "Total Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Motorbike Number : {:03d}".format(count_motorbike), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Car Number : {:03d}".format(count_car), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bicycle Number : {:03d}".format(count_bi), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bus Number : {:03d}".format(count_bus), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # cv2.putText(frame, "Train Number : {:03d}".format(count_train), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
        #             (0, 0, 255), 2)
        cv2.putText(frame, "Truck Number : {:03d}".format(count_truck), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw lane
        # cv2.line(frame, (START_POINT_L, 0), (END_POINT_L, width), (204, 255, 208), 2)
        # Draw start line
        cv2.line(frame, (0, START_POINT_HP1), (600, START_POINT_HP1), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (0, height - END_POINT_HP1), (600, height - END_POINT_HP1), (255, 0, 0), 2)
        # Show frame
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
        cv2.imshow("Counting", frame)
        # return frame
    cap.release()
    out.release()
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def check_location1(box_x, box_width, width):
    """
    Check center point of object that passing end line or not
    :param box_x: y value of bounding box
    :param box_width: height of bounding box
    :param width: height of image
    :return: Boolean
    """
    center_y = int(box_x + box_width / 2.0)
    if center_y > width - END_POINT_W:
        return True
    else:
        return False


def check_start_line1(box_x, box_width):
    """
    Check center point of object that passing start line or not
    :param box_x: y value of bounding box
    :param box_width: height of bounding box
    :return: Boolean
    """
    center_y = int(box_x + box_width / 2.0)
    if center_y > START_POINT_W:
        return True
    else:
        return False

def counting_vehicle1(skip_frame=1):
    global file_Name, cap

    colors = np.random.uniform(0, 255, size=(len(CLASSES), 3))

    # Load yolo model
    net = cv2.dnn.readNetFromDarknet(YOLOV4_CFG, YOLOV4_WEIGHT)

    # Read first frame
    # cap = cv2.VideoCapture(video_input)
    ret_val, frame = cap.read()
    width = frame.shape[1]
    height = frame.shape[0]

    # Define format of output
    video_format = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('vehicles.avi', video_format, 25, (width, height))

    # Define tracking object
    list_object = []
    number_frame = 0
    number_vehicle = 0
    count_car = 0
    count_bi = 0
    count_motorbike = 0
    count_bus = 0
    count_train = 0
    count_truck = 0
    while cap.isOpened():
        number_frame += 1
        # Read frame
        ret_val, frame = cap.read()
        if frame is None:
            break
        # Tracking old object
        tmp_list_object = list_object
        list_object = []
        for obj in tmp_list_object:
            tracker = obj['tracker']
            class_id = obj['id']
            confidence = obj['confidence']
            check, box = tracker.update(frame)
            if check:
                box_x, box_y, box_width, box_height = box
                draw_prediction_h(CLASSES, colors, frame, class_id, confidence,
                                box_x, box_y, box_width, box_height, speed[idx])
                obj['tracker'] = tracker
                obj['box'] = box
                if check_location1(box_x, box_width, width):
                    # This object passed the end line
                    number_vehicle += 1
                    if class_id == 1:
                        count_bi += 1
                    if class_id == 2:
                        count_car += 1
                    if class_id == 3:
                        count_motorbike += 1
                    if class_id == 5:
                        count_bus += 1
                    if class_id == 6:
                        count_train += 1
                    if class_id == 7:
                        count_truck += 1
                    cv2.line(frame, (width - END_POINT_W, 0), (width - END_POINT_W, width), (0, 0, 255), 3)
                else:
                    list_object.append(obj)

        if number_frame % skip_frame == 0:
            # Detect object and check new object
            boxes, class_ids, confidences = detections_yolo4(net, frame, CONFIDENCE_SETTING, YOLOV4_WIDTH,
                                                             YOLOV4_HEIGHT, width, height, classes=CLASSES)
            for idx, box in enumerate(boxes):
                box_x, box_y, box_width, box_height = box
                if not check_location1(box_x, box_width, width):
                    # This object doesnt pass the end line
                    box_center_x = int(box_x + box_width / 2.0)
                    box_center_y = int(box_y + box_height / 2.0)
                    StartPosition[idx] = box_x, box_y, box_width, box_height
                    check_new_object = True
                    for tracker in list_object:
                        # Check exist object
                        current_box_x, current_box_y, current_box_width, current_box_height = tracker['box']
                        current_box_center_x = int(current_box_x + current_box_width / 2.0)
                        current_box_center_y = int(current_box_y + current_box_height / 2.0)
                        # Calculate distance between 2 object
                        distance = math.sqrt((box_center_x - current_box_center_x) ** 2 +
                                             (box_center_y - current_box_center_y) ** 2)
                        CurrentPosition[idx] = [current_box_x, current_box_y, current_box_width,
                                                current_box_height]
                        # endtime = time.time() - start_time
                        # fps = number_frame / endtime
                        endtime = time.time()
                        fps = 1 / (endtime - start_time)
                        # track and speed calculation
                        [x1, y1, w1, h1] = StartPosition[idx]
                        [x2, y2, w2, h2] = CurrentPosition[idx]

                        StartPosition[idx] = [x2, y2, w2, h2]
                        # moving object
                        if [x1, y1, w1, h1] != [x2, y2, w2, h2]:
                            speed[idx] = calculate_speed([x1, y1, w1, h1], [x2, y2, w2, h2], fps)
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                        if distance < MAX_DISTANCE:
                            # Object is existed
                            check_new_object = False
                            break
                    if check_new_object and check_start_line1(box_x, box_width):
                        # Append new object to list
                        new_tracker = cv2.legacy.TrackerKCF_create()
                        new_tracker.init(frame, tuple(box))
                        new_object = {
                            'id': class_ids[idx],
                            'tracker': new_tracker,
                            'confidence': confidences[idx],
                            'box': box
                        }

                        list_object.append(new_object)
                        # Draw new object
                        draw_prediction_h(CLASSES, colors, frame, new_object['id'], new_object['confidence'],
                                        box_x, box_y, box_width, box_height, speed[idx])
        # FPS
        endtime = time.time() - start_time
        fps = number_frame / endtime

        cv2.putText(frame, "FPS: " + str(round(fps, 2)), (1200, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Put summary text
        cv2.putText(frame, "Total Number : {:03d}".format(number_vehicle), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Motorbike Number : {:03d}".format(count_motorbike), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Car Number : {:03d}".format(count_car), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bicycle Number : {:03d}".format(count_bi), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Bus Number : {:03d}".format(count_bus), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        cv2.putText(frame, "Truck Number : {:03d}".format(count_truck), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 0, 255), 2)
        # Draw start line
        cv2.line(frame, (START_POINT_W, 0), (START_POINT_W, width), (204, 90, 208), 1)
        # Draw end line
        cv2.line(frame, (width - END_POINT_W, 0), (width - END_POINT_W, width), (255, 0, 0), 2)
        # Show frame
        cv2.imshow("Counting", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        out.write(frame)
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def Frame1():
    # cap = cv2.VideoCapture('videos/1.mp4')
    ret, frame = cap.read()
    if ret == True:
        frame = imutils.resize(frame, width=1048)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        im = Image.fromarray(frame)
        img = ImageTk.PhotoImage(image=im)
        lblVideo_On.configure(image=img)
        lblVideo_On.image = img
        root.after(15, Frame1)
    else:
        lblVideo.configure(text="Disconnect to video")
        lblVideo_On.image = ""
        cap.release()
def Frame():
    global cap
    if cap is not None:
        ret, frame = cap.read()
        if ret == True:
            frame = imutils.resize(frame, width=1048)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            im = Image.fromarray(frame)
            img = ImageTk.PhotoImage(image=im)
            lblVideo_On.configure(image=img)
            lblVideo_On.image = img
            root.after(15, Frame)
        else:
            lblVideo.configure(text="Disconnect to video")
            lblVideo_On.image = ""
            cap.release()
def connect():
    global cap, file_Name
    if cap is not None:
        lblVideo_On.image = ""
        cap.release()
        cap = None
    file_Name = filedialog.askopenfilename(filetypes=[("all video format", ".mp4"), ("all video format", ".avi")])
    if len(file_Name) > 0:
        lblVideo.configure(text=file_Name)
        cap = cv2.VideoCapture(file_Name)
        Frame()
    else:
        lblVideo.configure(text="Disconnect to video")

def connect1():
    global cap, file_Name
    if cap is not None:
        lblVideo_On.image = ""
        cap.release()
        cap = None
    cap = cv2.VideoCapture("vehicles.avi")
    Frame1()

cap = None
file_Name = ""
root = Tk()
root.title("Object Detection")
root.configure(bg="#222222")
w_value = root.winfo_screenwidth()
h_value = root.winfo_screenheight()
# root.geometry("%dx%d+0+0" % (w_value, h_value))
root.resizable(False, False)
# root.resizable(True, True)

# root.columnconfigure(3, weight=1)
# root.rowconfigure(3, weight=1)
#
# my_size = ttk.Sizegrip(root)
# my_size.grid(row=3, sticky=SE)
# my_Frame = tk.Frame(root, highlightbackground="gray", highlightthickness=1)
# my_Frame.pack(side="bottom", fill=X)
# my_Frame.grid(column=0, row=3, columnspan=6)
lblVideo = tk.Label(root, text="Disconnect to video", font=("Arial", 13), bg="#222222", fg="white")
lblVideo.grid(column=0, row=1, columnspan=6)
# lblVideo.place(x=250, y=40)
lblVideo_On = tk.Label(root, bg="#222222")
lblVideo_On.grid(column=0, row=2, columnspan=6)
# lblVideo_On.place(x=10, y=60)
def btnHover(e):
    btnConnect["bg"] = "#444444"

def btnHover_L(e):
    btnConnect["bg"] = "#222222"

def btnHover0(e):
    btnConnect0["bg"] = "#444444"

def btnHover_L0(e):
    btnConnect0["bg"] = "#222222"

def btnHover1(e):
    btnConnect1["bg"] = "#444444"

def btnHover_L1(e):
    btnConnect1["bg"] = "#222222"

def btnHover2(e):
    btnConnect2["bg"] = "#444444"

def btnHover_L2(e):
    btnConnect2["bg"] = "#222222"

def btnHover3(e):
    btnConnect3["bg"] = "#444444"

def btnHover_L3(e):
    btnConnect3["bg"] = "#222222"

def btnHover4(e):
    btnConnect4["bg"] = "#444444"

def btnHover_L4(e):
    btnConnect4["bg"] = "#222222"

def btnHover5(e):
    btn["bg"] = "#444444"

def btnHover_L5(e):
    btn["bg"] = "#222222"

btnConnect = tk.Button(root, text = "Connect to video", font=("Arial", 12), command=connect, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect.grid(column=0, row=0, padx=3, pady=3)
# btnConnect.place(x=10, y=10)
btnConnect0 = tk.Button(root, text="DT_Normal", font=("Arial", 12), command=counting_vehicle2, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect0.grid(column=1, row=0, padx=3, pady=3)
# btnConnect0.place(x=150, y=10)
btnConnect1 = tk.Button(root, text="DT_Vertical", font=("Arial", 12), command=counting_vehicle, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect1.grid(column=2, row=0, padx=3, pady=3)
# btnConnect1.place(x=250, y=10)
btnConnect2 = tk.Button(root, text="DT_Horizontal", font=("Arial", 12), command=counting_vehicle1, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect2.grid(column=3, row=0, padx=3, pady=3)
# btnConnect2.place(x=360, y=10)
btnConnect3 = tk.Button(root, text="HW_In", font=("Arial", 12), command=counting_vehicle3, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect3.grid(column=4, row=0, padx=3, pady=3)
# btnConnect3.place(x=480, y=10)
btnConnect4 = tk.Button(root, text="HW_Out", font=("Arial", 12), command=counting_vehicle4, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btnConnect4.grid(column=5, row=0, padx=3, pady=3)
# btnConnect4.place(x=560, y=10)
btn = tk.Button(root, text="Video Detected", font=("Arial", 12),command=connect1, borderwidth=0, bg="#222222"
                       , fg="white", activebackground="#CCCCCC", activeforeground="black")
btn.grid(column=0, row=4, columnspan=6)

btnConnect.bind("<Enter>", btnHover)
btnConnect.bind("<Leave>", btnHover_L)
btnConnect0.bind("<Enter>", btnHover0)
btnConnect0.bind("<Leave>", btnHover_L0)
btnConnect1.bind("<Enter>", btnHover1)
btnConnect1.bind("<Leave>", btnHover_L1)
btnConnect2.bind("<Enter>", btnHover2)
btnConnect2.bind("<Leave>", btnHover_L2)
btnConnect3.bind("<Enter>", btnHover3)
btnConnect3.bind("<Leave>", btnHover_L3)
btnConnect4.bind("<Enter>", btnHover4)
btnConnect4.bind("<Leave>", btnHover_L4)
btn.bind("<Enter>", btnHover5)
btn.bind("<Leave>", btnHover_L5)
if __name__ == '__main__':
    root.mainloop()

