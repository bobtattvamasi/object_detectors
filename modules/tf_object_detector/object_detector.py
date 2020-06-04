# coding: utf-8
import copy

import tensorflow as tf
import numpy as np
import cv2
#import pycuda.autoinit  # This is needed for initializing CUDA driver
import time

from modules.tf_object_detector.utils.misc_utils import parse_anchors, read_class_names
from modules.tf_object_detector.utils.nms_utils import gpu_nms
from modules.tf_object_detector.utils.plot_utils import get_color_table, plot_one_box
from modules_helper.deep_sort_tracker_helper.deep_sort_tracker_helper import\
    deep_sort_tracker_helper
from modules_helper.deep_sort_tracker_helper.deep_sort.track import\
    DirectMoveState, TrackState

from modules.tf_object_detector.model import yolov3

from drawer.DrawScript import *
from drawer.Drawer import *

# import warnings filter
from warnings import simplefilter
# ignore all deprecation warnings
simplefilter(action='ignore', category=DeprecationWarning)


class object_detector:
    """docstring for object_detector"""

    def __init__(self):
        print('[Module][object_detector]: Init')

        self.tracker_lines = {}
        self.tracker_helpers = {}

        self.input_size = [416, 416]
        model_path = "modules/tf_object_detector/data/darknet_weights/yolov3.ckpt"
        anchors_path = "modules/tf_object_detector/data/yolo_anchors.txt"
        labels_path = "modules/tf_object_detector/data/coco.names"

        anchors = parse_anchors(anchors_path)
        self.classes = read_class_names(labels_path)
        num_class = len(self.classes)

        config = tf.ConfigProto()
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

        self.sess = tf.Session(config=config)
        self.input_data = tf.placeholder(tf.float32, [1, self.input_size[1],
                self.input_size[0], 3], name='input_data')
        yolo_model = yolov3(num_class, anchors)

        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(self.input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        self.boxes, self.scores, self.labels = gpu_nms(pred_boxes, pred_scores,
                num_class, max_boxes=100, score_thresh=0.7, iou_thresh=0.5)

        self.saver = tf.train.Saver()
        self.saver.restore(self.sess, model_path)
        self.regions = []
        self.colors = [ (255, 255, 255), (0, 255, 255), (0, 0, 255), (255, 0, 255), (255, 255, 0), (127, 21, 173), (222, 230, 31), (4, 71, 252),
                        (131, 243, 96), (91, 240, 83), (28, 185, 175), (147, 147, 43), (74, 27, 51), (246, 208, 101), (122, 199, 200), (2, 198, 8),
                        (164, 199, 229), (58, 249, 131), (211, 81, 10), (153, 174, 136), (45, 254, 224), (80, 113, 77), (70, 243, 38), (61, 237, 61), (188, 90, 221),
                        (67, 48, 57), (224, 37, 54), (112, 172, 36), (28, 34, 63), (126, 187, 65), (206, 146, 136), (207, 145, 223), (52, 225, 238), (92, 20, 58),
                        (23, 203, 91), (84, 97, 204), (49, 201, 238), (182, 73, 121), (166, 138, 149), (199, 106, 9), (123, 218, 135), (51, 204, 171), (128, 25, 70),
                        (93, 53, 224), (163, 102, 193), (78, 36, 146), (115, 25, 218), (204, 164, 135), (162, 249, 15), (8, 41, 0), (142, 55, 73), (168, 254, 246),
                        (29, 236, 46), (8, 91, 254), (247, 250, 29), (13, 225, 200), (238, 211, 112), (81, 242, 180), (111, 246, 73), (211, 220, 148), (230, 238, 186),
                        (186, 186, 73), (53, 247, 82), (49, 101, 45), (219, 24, 146), (227, 32, 242), (119, 19, 235), (190, 220, 68), (13, 127, 195), (140, 22, 37),
                        (21, 249, 203), (140, 161, 225), (193, 23, 82), (188, 11, 132), (57, 38, 218), (181, 188, 53), (217, 107, 1), (222, 167, 91), (13, 251, 188), (114, 70, 37)]
    def process(self, frame):
        raw_frame = frame
        module_settings = module_settings = {
    "_id": "5ddfc1fec7663c00103a38bb",
    "boxes": 1,
    "confidence": 0.7,
    "direction_regions": [],
    "draw_confid": 0,
    "draw_name": 1,
    "draw_settings": {
        "_id": "5ddfc1fec7663c00103a38bc",
        "color": [
            255,
            140,
            0
        ],
        "font_size": 1,
        "line_thickness": 2
    },
    "is_active": 0,
    "lines": [],
    "regions": [],
    "target_classes": [
        0
    ],
    "tracker_mode": 0
}

        draw_script = DrawScript()
        drawer = Drawer()

        # classes_to_detect
        classes_to_detect = module_settings['target_classes']
        # confidence_level
        confidence_level = module_settings['confidence']

        ms, fps = 0, 0

        if len(classes_to_detect) <= 0 or confidence_level > 1:
            return draw_script, ms, fps

        resized = np.zeros((416, 416, 3), np.uint8)
        cv2.resize(raw_frame, (416, 416), dst=resized,
                interpolation=cv2.INTER_LINEAR)
        converted = np.zeros((416, 416, 3), np.uint8)
        cv2.cvtColor(resized, cv2.COLOR_BGR2RGB, dst=converted)
        converted_as_float = np.asarray(converted, np.float32)
        converted_as_float[np.newaxis, :] /= 255.
        
        # Инференс йолки
        time_start = time.time()
        try:
            boxes_, scores_, labels_ = self.sess.run([self.boxes, self.scores,
                    self.labels], feed_dict={self.input_data: [converted_as_float]})
        except:
            return draw_script, ms, fps
        time_end = time.time()
        
        ms = int((time_end - time_start)*1000)
        fps = int(1000/ms)

        # rescale the coordinates to the original image
        height, width = raw_frame.shape[:2]
        should_draw_trail = True
        cam_id = "5de009e28b73200010185f59"

        eff_boxes = []
        eff_labels = []
        eff_scores = []

        try:
            for i in range(len(boxes_)):
                if labels_[i] in classes_to_detect and\
                        scores_[i] > confidence_level:
                    eff_boxes.append([
                        int(boxes_[i][0] / float(self.input_size[0]) * width),
                        int(boxes_[i][1] / float(self.input_size[1]) * height),
                        int(boxes_[i][2] / float(self.input_size[0]) * width),
                        int(boxes_[i][3] / float(self.input_size[1]) * height)
                    ])
                    eff_boxes[i][2] -= eff_boxes[i][0]
                    eff_boxes[i][3] -= eff_boxes[i][1]

                    eff_labels.append(self.classes[labels_[i]])
                    eff_scores.append(scores_[i])
        except Exception:
            print('[Module][object_detector]: process frame error (180line)')
            return draw_script, ms, fps

        if cam_id not in self.tracker_helpers:
            self.tracker_helpers[cam_id] = deep_sort_tracker_helper()
        self.tracker_helpers[cam_id].track(np.array(eff_boxes), eff_scores,
                eff_labels, module_settings, frame)

        # Создадим записи для статистики ClickHouse для каждого нового объекта
        for track in self.tracker_helpers[cam_id].tracker.tracks:
            if track.is_new and track.state == TrackState.Confirmed:
                track.is_new = False       

        for track in self.tracker_helpers[cam_id].tracker.tracks:
            if not track.should_be_drawn():
                continue

            if track.class_ in list(self.classes.values()):
                color = self.colors[list(self.classes.values()).index(track.class_)] #[eff_labels.index(track.class_)]]
            else:
                color = None

            if should_draw_trail:
                if track.direct_move_state == DirectMoveState.Yellow:
                    color = (0, 240, 255)
                if track.is_in_forbidden_region or track.direct_move_state == DirectMoveState.Red:
                    color = (0, 0, 200)

                prev_point = None
                for point in track.history:
                    draw_script.add_circle(Circle(point, 5, color=color))
                    if prev_point is not None:
                        draw_script.add_line(
                            Line(point, prev_point, color=color))
                    prev_point = point
                last_pos = track.get_last_position()
                if last_pos is not None and len(track.history) > 0:
                    draw_script.add_line(
                        Line(last_pos, track.history[0], color=color))
                    draw_script.add_circle(Circle(last_pos, 5, color=color))

            box = track.to_tlbr().astype(np.int32)
            draw_script.add_box(Box((box[0], box[1]), (box[2], box[3]),
                    color=color))
            if module_settings['draw_name'] == 1:
                label = track.class_
                if module_settings['draw_confid'] == 1:
                    label = '{} {:.2f}%'.format(track.class_,
                            track.confidence * 100)
                draw_script.add_label(Label(label, (box[0] + 10, box[1] + 35),
                        color=color))
        return draw_script, ms, fps

    def close(self):
        self.sess.close()
