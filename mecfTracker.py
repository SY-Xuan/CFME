from kcftracker import KCFTracker 
import cv2
import numpy as np

class mecfTracker():
    def __init__(self, hog=True, fixed_window=True, scale=False, occlusion_threshold=0.3):
        self.tracker = KCFTracker(hog=hog, fixed_window=fixed_window, multiscale=scale, occlusion_threshold=occlusion_threshold)
        self.kalman = cv2.KalmanFilter(4,2)
        self.kalman.measurementMatrix = np.array([[1,0,0,0],[0,1,0,0]],np.float32)
        self.kalman.transitionMatrix = np.array([[1,0,1,0],[0,1,0,1],[0,0,1,0],[0,0,0,1]], np.float32)
        self.kalman.processNoiseCov = np.array([[1,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]], np.float32) * 0.001
        self.kalman.measurementNoiseCov = np.array([[1,0],[0,1]], np.float32) * 1
        self.trace_array = []
        self.predict = [0, 0]
        # whether kalman filter can be used
        self.iskalman_work = False
        # whether the object is occluded
        self.isocclution = False

        self.occlution_index = 0
        self.tem_trace_array = []
        self.frame_index = 0
        self.last_bbox = []
        self.kalman_work_num = 0
        self.occlusion_threshold = occlusion_threshold

    def init(self, frame, bbox):
        self.trace_array.append(bbox)
        self.kalman.correct(np.array([[np.float32(bbox[0])],[np.float32(bbox[1])]]))
        self.tracker.init([bbox[0],bbox[1],bbox[2],bbox[3]], frame)
        self.frame_index += 1

    def update(self, frame):
        if self.iskalman_work:	
            next_bbox = [self.predict[0], self.predict[1], self.last_bbox[2], self.last_bbox[3]]
            self.last_bbox, peak_value = self.tracker.update(frame, next_bbox, isUse=True)
            # long-term
            if peak_value > self.occlusion_threshold:
                self.trace_array.append(self.last_bbox.copy())
                self.kalman.correct(np.array([[np.float32(self.last_bbox[0])],[np.float32(self.last_bbox[1])]]))
                self.predict = self.kalman.predict()
                
            else:
                self.last_bbox = [next_bbox[0], next_bbox[1], self.last_bbox[2], self.last_bbox[3]]
                self.predict = self.kalman.predict()
        else:
            if len(self.trace_array) > 4:
                dx = 0
                dy = 0
                for i in range(-5, -1):
                    dx += self.trace_array[i + 1][0] - self.trace_array[i][0]
                    dy += self.trace_array[i + 1][1] - self.trace_array[i][1]
                next_bbox = [self.last_bbox[0] + dx / 4, self.last_bbox[1] + dy / 4, self.last_bbox[2], self.last_bbox[3]]
                self.last_bbox, peak_value = self.tracker.update(frame, next_bbox, isUse=True)
                # long-term
                if peak_value < 0.3:
                    self.last_bbox = [next_bbox[0], next_bbox[1], self.last_bbox[2], self.last_bbox[3]]
                    self.isocclution = True
                else:
                    if self.isocclution == True:
                        if self.occlution_index != 0:
                            self.tem_trace_array.append(self.last_bbox.copy())
                        self.occlution_index += 1
                        
                        if self.occlution_index == 6:
                            
                            self.trace_array.extend(self.tem_trace_array)
                            self.isocclution = False
                    else:
                        self.trace_array.append(self.last_bbox.copy())
                
            else:
                self.last_bbox, peak_value = self.tracker.update(frame)
                self.trace_array.append(self.last_bbox.copy())
                
            
            if (abs(self.predict[0] - self.last_bbox[0]) < 2) and (abs(self.predict[1] - self.last_bbox[1]) < 2):
                self.kalman_work_num += 1
                if self.kalman_work_num == 3:
                    self.iskalman_work = True
                    self.kalman_work_num = 0
            else:
                self.kalman_work_num = 0
            self.kalman.correct(np.array([[np.float32(self.last_bbox[0])],[np.float32(self.last_bbox[1])]]))
            self.predict = self.kalman.predict()
        self.frame_index += 1
        return True, self.last_bbox
