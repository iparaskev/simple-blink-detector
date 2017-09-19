import cv2
import dlib
import numpy as np
from keras.models import load_model
from scipy.spatial import distance as dist
from imutils import face_utils

predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
face_cascade = cv2.CascadeClassifier('haarcascade/haarcascade_frontalface_alt.xml')

# detect the face rectangle 
def detect(img, cascade = face_cascade , minimumFeatureSize=(20, 20)):
    if cascade.empty():
        raise (Exception("There was a problem loading your Haar Cascade xml file."))
    rects = cascade.detectMultiScale(img, scaleFactor=1.3, minNeighbors=1, minSize=minimumFeatureSize)
    
    # if it doesn't return rectangle return array
    # with zero lenght
    if len(rects) == 0:
        return []

    #  convert last coord from (width,height) to (maxX, maxY)
    rects[:, 2:] += rects[:, :2]

    return rects

def cropEyes(frame):
	 
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	# detect the face at grayscale image
	te = detect(gray, minimumFeatureSize=(80, 80))

	# if the face detector doesn't detect face
	# return None, else if detects more than one faces
	# keep the bigger and if it is only one keep one dim
	if len(te) == 0:
		return None
	elif len(te) > 1:
		face = te[0]
	elif len(te) == 1:
		[face] = te

	# keep the face region from the whole frame
	face_rect = dlib.rectangle(left = int(face[0]), top = int(face[1]),
								right = int(face[2]), bottom = int(face[3]))
	
	# determine the facial landmarks for the face region
	shape = predictor(gray, face_rect)
	shape = face_utils.shape_to_np(shape)

	#  grab the indexes of the facial landmarks for the left and
	#  right eye, respectively
	(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
	(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

	# extract the left and right eye coordinates
	leftEye = shape[lStart:lEnd]
	rightEye = shape[rStart:rEnd]

	# keep the upper and the lower limit of the eye 
	# and compute the height 
	l_uppery = min(leftEye[1:3,1])
	l_lowy = max(leftEye[4:,1])
	l_dify = abs(l_uppery - l_lowy)

	# compute the width of the eye
	lw = (leftEye[3][0] - leftEye[0][0])

	# we want the image for the cnn to be (26,34)
	# so we add the half of the difference at x and y
	# axis from the width at height respectively left-right
	# and up-down 
	minxl = int(round(leftEye[0][0] - ((34-lw)/2)))
	maxxl = int(round(leftEye[3][0] + ((34-lw)/2))) 
	minyl = int(round(l_uppery - ((26-l_dify)/2)))
	maxyl = int(round(l_lowy + ((26-l_dify)/2)))
	
	# crop the eye rectangle from the frame
	left_eye_rect = [minxl, minyl, maxxl, maxyl]
	left_eye_image = gray[int(np.round(left_eye_rect[1])):left_eye_rect[3], int(left_eye_rect[0]):left_eye_rect[2]]
	
	# same as left eye at right eye
	r_uppery = min(rightEye[1:3,1])
	r_lowy = max(rightEye[4:,1])
	r_dify = abs(r_uppery - r_lowy)
	rw = int(round(rightEye[3][0] - rightEye[0][0]))
	minxr = int(round(rightEye[0][0]-((34-rw)/2)))
	maxxr = int(round(rightEye[3][0] + ((34-rw)/2)))
	minyr = int(round(r_uppery - ((26-r_dify)/2)))
	maxyr = int(round(r_lowy + ((26-r_dify)/2)))
	right_eye_rect = [minxr, minyr, maxxr, maxyr]
	right_eye_image = gray[right_eye_rect[1]:right_eye_rect[3], right_eye_rect[0]:right_eye_rect[2]]

	# if it doesn't detect left or right eye return None
	if 0 in left_eye_image.shape or 0 in right_eye_image.shape:
		return None
	# resize for the conv net
	left_eye_image = cv2.resize(left_eye_image, (34, 26))
	right_eye_image = cv2.resize(right_eye_image, (34, 26))
	right_eye_image = cv2.flip(right_eye_image, 1)
	# return left and right eye
	return left_eye_image, right_eye_image 

# make the image to have the same format as at training 
def cnnPreprocess(img):
	img = img.astype('float32')
	img /= 255
	img = np.expand_dims(img, axis=2)
	img = np.expand_dims(img, axis=0)
	return img

def main():
	camera = cv2.VideoCapture(0)
	model = load_model('blinkModel_aug.hdf5')
	close_counter = blinks = mem_counter= 0
	state = ''
	while True:
		
		ret, frame = camera.read()
		
		# detect eyes
		eyes = cropEyes(frame)
		if eyes is None:
			continue
		else:
			left_eye,right_eye = eyes
		
		# average the predictions of the two eyes 
		prediction = (model.predict(cnnPreprocess(left_eye)) + model.predict(cnnPreprocess(right_eye)))/2.0
			
		# blinks
		# if the eyes are open reset the counter for close eyes
		if prediction > 0.5 :
			state = 'open'
			close_counter = 0
		else:
			state = 'close'
			close_counter += 1
		
		# if the eyes are open and previousle were closed
		# for sufficient number of frames then increcement 
		# the total blinks
		if state == 'open' and mem_counter > 1:
			blinks += 1
		# keep the counter for the next loop 
		mem_counter = close_counter 

		# draw the total number of blinks on the frame along with
		# the state for the frame
		cv2.putText(frame, "Blinks: {}".format(blinks), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		cv2.putText(frame, "State: {}".format(state), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
		
		# show the frame
		cv2.imshow('blinks counter', frame)
		key = cv2.waitKey(1) & 0xFF

		# if the `q` key was pressed, break from the loop
		if key == ord('q'):
			break
	# do a little clean up
	cv2.destroyAllWindows()
	del(camera)


if __name__ == '__main__':
	main()