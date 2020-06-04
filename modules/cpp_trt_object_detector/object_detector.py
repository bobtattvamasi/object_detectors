# coding: utf-8
import copy


from modules.cpp_trt_object_detector.build.pydetector import Config, Detector, Result


import numpy as np
import cv2
import pycuda.autoinit  # This is needed for initializing CUDA driver
import time

from modules_helper.deep_sort_tracker_helper.deep_sort_tracker_helper import\
	deep_sort_tracker_helper
from modules_helper.deep_sort_tracker_helper.deep_sort.track import\
	DirectMoveState, TrackState



from drawer.DrawScript import *
from drawer.Drawer import *

from warnings import simplefilter
# ignore all deprecation warnings
simplefilter(action='ignore', category=DeprecationWarning)

from modules.trt_object_detector.utils.yolov3_classes import get_cls_dict


class object_detector:
	"""docstring for object_detector"""

	def __init__(self):
		print('[Module][object_detector]: Init')

		self.tracker_lines = {}
		self.tracker_helpers = {}
		
		self.classes = get_cls_dict('coco')
		YOLO_CONFIG = Config()

		YOLO_CONFIG.file_model_cfg = "modules/cpp_trt_object_detector/configs/yolov3.cfg"
		YOLO_CONFIG.file_model_weights = "modules/cpp_trt_object_detector/configs/yolov3.weights"
		YOLO_CONFIG.detect_thresh = 0.9
		YOLO_CONFIG.gpu_id = 0
		YOLO_CONFIG.net_type = Config.YOLOV3
		YOLO_CONFIG.precision = Config.FP32
		YOLO_CONFIG.log_level = Config.INFO
		YOLO_CONFIG.max_workspace_size = 1 << 30

		self.yolo = Detector()
		self.yolo.init(YOLO_CONFIG)

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
		raw_frame = frame.copy()
		module_settings = {"_id": "5ddfc1fec7663c00103a38bb",
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

		# Инференс йолки
		time_start = time.time()
		results = self.yolo.detect(raw_frame)
		time_end = time.time()
		boxes_, scores_, labels_ = [],[],[]
		for res in results:
			 boxes_.append(res.rect)
			 scores_.append(res.prob)
			 labels_.append(res.id)

		ms = int((time_end - time_start)*1000)
		fps = int(1000/ms)

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
						int(boxes_[i][0]),
						int(boxes_[i][1]),
						int(boxes_[i][2]),
						int(boxes_[i][3])
					])

					eff_labels.append(self.classes[labels_[i]])
					eff_scores.append(scores_[i])
		except Exception as e:
			print('[Module][object_detector]: process frame error (180line)')
			print(e)
			return draw_script

		# Закидываем инфу в трекер
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
		print("it is closed")