from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import io
import re
import time
import screeninfo
import os

from pathlib import Path

import numpy as np
from PIL import Image

from tflite_runtime.interpreter import Interpreter
import cv2

TEMP_FILE_PATH = '/sys/class/thermal/thermal_zone0/temp'


def load_labels(path):
    """Loads the labels file. Supports files with or without index numbers."""
    with open(path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        labels = {}
        for row_number, content in enumerate(lines):
            pair = re.split(r'[:\s]+', content.strip(), maxsplit=1)
            if len(pair) == 2 and pair[0].strip().isdigit():
                labels[int(pair[0])] = pair[1].strip()
            else:
                labels[row_number] = pair[0].strip()
    return labels


def set_input_tensor(interpreter, image):
    """Sets the input tensor."""
    tensor_index = interpreter.get_input_details()[0]['index']
    input_tensor = interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image


def get_output_tensor(interpreter, index):
    """Returns the output tensor at the given index."""
    output_details = interpreter.get_output_details()[index]
    # print(output_details)
    tensor = np.squeeze(interpreter.get_tensor(output_details['index']))
    # print(tensor)
    return tensor


def detect_objects(interpreter, image, threshold):
    """Returns a list of detection results, each a dictionary of object info."""
    set_input_tensor(interpreter, image)
    interpreter.invoke()

    # Get all output details
    boxes = get_output_tensor(interpreter, 0)
    # print("Boxes:",boxes)
    classes = get_output_tensor(interpreter, 1)
    # print("Classes",classes)
    scores = get_output_tensor(interpreter, 2)
    # print("Scores",scores)
    count = int(get_output_tensor(interpreter, 3))
    # print("Count",count)

    results = []
    for i in range(count):
        if scores[i] >= threshold:
            result = {
                'bounding_box': boxes[i],
                'class_id': classes[i],
                'score': scores[i]
            }
            results.append(result)
    return results


def annotate_objects(frame, results, labels, elapsed, stream_input_width, stream_input_height):
    """Draws the bounding box and label for each object in the results."""
    font = cv2.FONT_HERSHEY_DUPLEX

    if elapsed > 0:
        fps = '{0:.2f}'.format(1/(elapsed/1000))
        cv2.putText(frame, fps, (20, 20), font, 1.0, (255, 255, 0), 1)

    for obj in results:
        # Convert the bounding box figures from relative coordinates
        # to absolute coordinates based on the original resolution
        ymin, xmin, ymax, xmax = obj['bounding_box']
        xmin = int(xmin * stream_input_width)
        xmax = int(xmax * stream_input_width)
        ymin = int(ymin * stream_input_height)
        ymax = int(ymax * stream_input_height)

        # Overlay the box, label, and score on the camera preview
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), (0, 0, 255), 2)

        cv2.rectangle(frame, (xmin, ymax - 35),
                      (xmax, ymax), (0, 0, 255), cv2.FILLED)
        name = str('%s %.2f' % (labels[obj['class_id']], obj['score']))
        cv2.putText(frame, name, (xmin + 6, ymax - 6),
                    font, 1.0, (255, 255, 255), 1)


def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--model', help='File path of .tflite file.', required=True)
    parser.add_argument(
        '--labels', help='File path of labels file.', required=True)
    parser.add_argument(
        '--threshold',
        help='Score threshold for detected objects.',
        required=False,
        type=float,
        default=0.6)
    parser.add_argument(
        '--input_stream', help='Input video stream', required=False, default=0)
    args = parser.parse_args()

    labels = load_labels(args.labels)
    interpreter = Interpreter(args.model, num_threads=3)
    interpreter.allocate_tensors()
    shape = interpreter.get_input_details()[0]['shape']
    print("NN Shape:", shape)
    _, input_height, input_width, _ = shape
    cap = cv2.VideoCapture()
    cap.open(args.input_stream)
    window_name = "window"
    screen_id = 0
    elapsed_ms = 0

    # Obtain screen size
    screen = screeninfo.get_monitors()[screen_id]
    width, height = screen.width, screen.height

    if not cap.isOpened():
        print("Cannot open camera")
        exit()

    # Capture frame
    ret, frame = cap.read()
    # if frame is read correctly ret is True
    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        exit()
    # Get frames size
    stream_input_height, stream_input_width = frame.shape[:2]

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()

        # if frame is read correctly ret is True
        if not ret:
            print("Can't receive frame (stream end?). Exiting ...")
            break

        cv2.namedWindow(window_name, cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty(
            window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

        start_time = time.monotonic()
        frame_analice = cv2.resize(frame, (input_width, input_height))
        results = detect_objects(interpreter, frame_analice, args.threshold)

        annotate_objects(frame, results, labels, elapsed_ms,
                         stream_input_width, stream_input_height)
        elapsed_ms = (time.monotonic() - start_time) * 1000

        # ----->> Log prints <<-----
        if True:
            os.system('clear')
            print(
                '___ [ RPI Object detection with TFLite ] _________________________')
            print()
            print("Used model:\t", Path(args.model).name)
            print()
            print('Process FPS:\t', '{0:.2f}'.format(1/(elapsed_ms/1000)))
            with open(TEMP_FILE_PATH) as fp:
                line = fp.readline()
                line = int(line) / 1000

                print("CPU Temp:\t", line, "°C")
            print()
            print("Input size:")
            print("\twidth:\t", stream_input_width)
            print("\theight:\t", stream_input_height)
            print()
            print("Output size:")
            print("\twidth:\t", width)
            print("\theight:\t", height)
            print()
            if len(results):
                obj_count = len(results)
            else:
                obj_count = 0
            print("Detected objects:", obj_count)
            print("Threshold:\t", args.threshold)
            print()

        # Display the resulting frame
        frame = cv2.resize(frame, (width, height))
        cv2.imshow(window_name, frame)
        if cv2.waitKey(1) == ord('q'):
            break

    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
