from modules.trt_object_detector.object_detector import object_detector as trt_object_detector
from modules.tf_object_detector.object_detector import object_detector as tf_object_detector
from modules.cpp_trt_object_detector.object_detector import object_detector as cpp_trt_object_detector
import cv2
import time
import numpy
# from threading import Thread
# import queue
from drawer.Drawer import *

import argparse

# Add arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--type_of_neuronNet",type=str, required=True, default="cpp_trt", help="type of object detector: trt, tf, cpp_trt")
args = vars(ap.parse_args())

class VideoRunner():
	def __init__(self):

		if args['type_of_neuronNet'] == 'tf':
			self.obj_det = tf_object_detector()
		elif args['type_of_neuronNet'] == 'trt':
			self.obj_det = trt_object_detector()
		elif args['type_of_neuronNet'] == 'cpp_trt':
			self.obj_det = cpp_trt_object_detector()
		else:
			raise ValueError("Type of object detector didn't selected. Please choose one of this argument: trt, tf, cpp_trt \n(-.-)' For example run: python main.py -n cpp_trt")

		self.input_video = 0

		self.cap = cv2.VideoCapture(self.input_video)

		_, self.frame = self.cap.read()

	def process(self):

		while(self.cap.isOpened()):
			_, self.frame = self.cap.read()
			if _:
				draw_script, ms, fps = self.obj_det.process(self.frame)
				
				drawer = Drawer()
				self.frame = drawer.process(self.frame, draw_script)

				cv2.putText(self.frame, f"INFERENCE TIME/FPS: {ms} ms/{fps} fps", (2, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2, cv2.FILLED)
				cv2.imshow("TensorTV+", self.frame)
				if cv2.waitKey(25) & 0xFF == ord('q'):
					break
			else:
				self.cap = cv2.VideoCapture(self.input_video)

		self.cap.release()
		cv2.destroyAllWindows()
		self.obj_det.close()


vr = VideoRunner()
vr.process()