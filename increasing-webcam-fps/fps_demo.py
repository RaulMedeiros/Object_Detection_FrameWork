# USAGE
# python fps_demo.py
# python fps_demo.py --display 1

# import the necessary packages
from __future__ import print_function
from imutils.video import WebcamVideoStream
from imutils.video import FPS
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-n", "--num-frames", type=int, default=100,
	help="# of frames to loop over for FPS test")
ap.add_argument("-d", "--display", type=int, default=-1,
	help="Whether or not frames should be displayed")
ap.add_argument("-s", "--source", type=str, default='0',
	help="device id, path to a video file or 'rtps' link from a IP camera")
args = vars(ap.parse_args())

# created a *threaded *video stream, allow the camera senor to warmup,
# and start the FPS counter
print("[INFO] sampling THREADED frames from webcam...")

if( len(args['source']) == 1 ):
	src = int(args['source'])

vs = WebcamVideoStream(src=src).start()
fps = FPS().start()

# loop over some frames...this time using the threaded stream
while (True):
	# grab the frame from the threaded video stream and process it
	src_frame = vs.read()
		
	def do_something(frame):
		frame = imutils.resize(frame, width=400)
		return frame

	out_frame = do_something(src_frame)

	# check to see if the frame should be displayed to our screen
	if args["display"] > 0:
		cv2.imshow("Frame", out_frame)
		key = cv2.waitKey(1) & 0xFF

	# update the FPS counter
	fps.update()

# stop the timer and display FPS information
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# do a bit of cleanup
cv2.destroyAllWindows()
vs.stop()