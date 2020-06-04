import cv2
import json

class Drawer:
	"""docstring for Drawer"""
	def __init__(self):
		pass
	# def process(self, frame, draw_script):
	# 	#for draw_script in output:
	# 	if len(draw_script.boxes) > 0:
	# 		frame = self.draw_boxes(frame, draw_script.boxes)
	# 	if len(draw_script.lines) > 0:
	# 		frame = self.draw_lines(frame, draw_script.lines)
	# 	if len(draw_script.labels) > 0:
	# 		frame = self.draw_labels(frame, draw_script.labels)
	# 	if len(draw_script.circles) > 0:
	# 		frame = self.draw_circles(frame, draw_script.circles)
	# 	if len(draw_script.arrows) > 0:
	# 		frame = self.draw_arrows(frame, draw_script.arrows)

	# 	return frame

	# def draw_boxes(self, frame, script):
	# 	for box in script:
	# 		cv2.rectangle(frame, (box['script'][0], box['script'][1]), (box['script'][2], box['script'][3]), box['color'], box['line_thickness'])

	# 	return frame

	# def draw_lines(self, frame, script):
	# 	for line in script:

	# 		for line_segment in line['script']:
	# 			cv2.line(frame, (line_segment[0][0], line_segment[0][1]), (line_segment[1][0], line_segment[1][1]), line['color'], line['line_thickness'])

	# 	return frame

	# def draw_labels(self, frame, script):
	# 	for label in script:
	# 		cv2.putText(frame, label['script'][0], (label['script'][1], label['script'][2]), label['font'] , label['font_size'], label['color'], label['line_thickness'])

	# 	return frame

	# def draw_circles(self, frame, script):
	# 	for circle in script:
	# 		cv2.circle(frame, (circle['script'][0], circle['script'][1]), circle['script'][2], circle['color'], circle['line_thickness'])

	# 	return frame

	# def draw_arrows(self, frame, script):
	# 	for arrow in script:
	# 		cv2.arrowedLine(frame, (arrow['script'][0], arrow['script'][1]), (arrow['script'][2], arrow['script'][3]), arrow['color'], arrow['line_thickness'], tipLength=0.2)

	# 	return frame
	def process(self, frame, draw_scripts):
		#print("DRAW_SCRIPT: ", draw_scripts)
		outputs = []
		if draw_scripts:
			outputs.append(draw_scripts)
			for draw_script in outputs:
				for line in draw_script.lines:
					cv2.line(frame, line.point1, line.point2,
							line.color or draw_script.color,
							line.line_thickness or draw_script.line_thickness)
				for arrow in draw_script.arrows:
					cv2.arrowedLine(frame, arrow.point1, arrow.point2,
							arrow.color or draw_script.color,
							arrow.line_thickness or draw_script.line_thickness)
				for box in draw_script.boxes:
					cv2.rectangle(frame, box.corner1, box.corner2,
							box.color or draw_script.color,
							box.line_thickness or draw_script.line_thickness)
				for circle in draw_script.circles:
					cv2.circle(frame, circle.center, circle.radius,
							circle.color or draw_script.color,
							circle.line_thickness or draw_script.line_thickness)
				for label in draw_script.labels:
					cv2.putText(frame, label.text, label.point,
							label.font or draw_script.font,
							label.font_size or draw_script.font_size,
							label.color or draw_script.color,
							label.line_thickness or draw_script.line_thickness,
							label.bottom_left_origin)

		return frame