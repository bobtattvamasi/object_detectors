import numpy as np

def apply_offsets(rectangle, offsets):
	x, y, width, height = rectangle[0], rectangle[3], rectangle[2], rectangle[1]
	x_off, y_off = offsets
	return (x - x_off, x + width + x_off, y - y_off, y + height + y_off)

def preprocess_input(x, v2=True):
	x = x.astype('float32')
	x = x / 255.0
	if v2:
		x = x - 0.5
		x = x * 2.0
	return x

def get_prediction_dataframe(prediction, i):
	if round(prediction.loc[i]['Male'], 2) > 0.7:
		sex = 'M'
	else:
		sex = 'F'

	if round(prediction.loc[i]['No Eyewear'], 2) > 0.7:
		glass = 'No'
	else:
		glass = 'Yes'
	race = np.argmax(prediction.loc[i][1:4])
	age = np.argmax(prediction.loc[i][5:9])

	return sex + " " + glass + " " + str(race)+ " " + str(age)

def resize(frame):
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
