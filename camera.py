import cv2
import threading
import time
import logging

logger = logging.getLogger(__name__)
thread = None

backSub = cv2.createBackgroundSubtractorMOG2(history=90, varThreshold=50,detectShadows = True)

def proccess(frame):
        copy = frame.copy()
        gray = cv2.cvtColor(copy, cv2.COLOR_BGR2GRAY)
        blur = cv2.blur(frame,(5,5))
        mask = backSub.apply(blur)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reverse = np.invert(mask)
        gray = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        masked1 = cv2.bitwise_and(frame,frame, mask=mask)
        masked2 = cv2.bitwise_and(gray,gray, mask=reverse)
        masked = cv2.addWeighted(masked1,1,masked2,.5,0)
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        for c in contours:
                x,y,w,h = cv2.boundingRect(c)
                if w>10 and h>10:
                        cv2.rectangle(frame,(x,y),(x+w,y+h),(155,255,0),2, cv2.LINE_AA)
                        cv2.rectangle(masked,(x,y),(x+w,y+h),(155,255,0),2, cv2.LINE_AA)
        out1 = cv2.vconcat([frame,mask])
        out2 = cv2.vconcat([gray,masked])
        out = cv2.hconcat([out1,out2])
	return out


class Camera:
	def __init__(self,fps=30,video_source='rtmp://x.x.x.x:1935/live'):
		logger.info(f"Initializing camera class with {fps} fps and video_source={video_source}")
		self.fps = fps
		self.video_source = video_source
		self.camera = cv2.VideoCapture(self.video_source)
		# We want a max of 5s history to be stored, thats 5s*fps
		self.max_frames = 5*self.fps
		self.frames = []
		self.isrunning = False
	def run(self):
		logging.debug("Perparing thread")
		global thread
		if thread is None:
			logging.debug("Creating thread")
			thread = threading.Thread(target=self._capture_loop,daemon=True)
			logger.debug("Starting thread")
			self.isrunning = True
			thread.start()
			logger.info("Thread started")

	def _capture_loop(self):
		dt = 1/self.fps
		logger.debug("Observation started")
		while self.isrunning:
			v,im = self.camera.read()
			im = proccess(im)

			if v:
				if len(self.frames)==self.max_frames:
					self.frames = self.frames[1:]
				self.frames.append(im)
			time.sleep(dt)
		logger.info("Thread stopped successfully")

	def stop(self):
		logger.debug("Stopping thread")
		self.isrunning = False

	def get_frame(self, _bytes=True):
		if len(self.frames)>0:
			if _bytes:
				img = cv2.imencode('.png',self.frames[-1])[1].tobytes()
			else:
				img = self.frames[-1]
		else:
			with open("images/not_found.jpeg","rb") as f:
				img = f.read()
		return img
		
