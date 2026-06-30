# import numpy as np
# import cv2 as cv
# import subprocess
# import time
# import os
# from ObjectDetection import detectObject, displayImage
# import sys
# from gtts import gTTS   
# from playsound import playsound
# import os
# from threading import Thread 

# global class_labels
# global cnn_model
# global cnn_layer_names
# playcount = 0

# def deleteDirectory():
#     filelist = [ f for f in os.listdir('play') if f.endswith(".mp3") ]
#     for f in filelist:
#         os.remove(os.path.join('play', f))

# def speak(data, playcount):
#     class PlayThread(Thread):
#         def __init__(self, data, playcount):
#             Thread.__init__(self) 
#             self.data = data
#             self.playcount = playcount
#         def run(self):
#             t1 = gTTS(text=self.data, lang='en', slow=False)
#             t1.save("play/"+str(self.playcount)+".mp3")
#             playsound("play/"+str(self.playcount)+".mp3")
            

#     newthread = PlayThread(data, playcount) 
#     newthread.start()

# def loadLibraries(): #function to load yolov3 model weight and class labels
#         global class_labels
#         global cnn_model
#         global cnn_layer_names
#         class_labels = open('model/yolov3-labels').read().strip().split('\n') #reading labels from yolov3 model
#         cnn_model = cv.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights') #reading model
#         cnn_layer_names = cnn_model.getLayerNames() #getting layers from cnn model
#         cnn_layer_names = [cnn_layer_names[i[0] - 1] for i in cnn_model.getUnconnectedOutLayers()] #assigning all layers

# def detectFromImage(imagename): #function to detect object from images
#         #random colors to assign unique color to each label
#         label_colors = np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
#         try:
#                 image = cv.imread(imagename) #image reading
#                 image_height, image_width = image.shape[:2] #converting image to two dimensional array
#         except:
#                 raise 'Invalid image path'
#         finally:
#                 image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels)#calling detection function
#                 displayImage(image)#display image with detected objects label

# def detectFromVideo(): #function to read objects from video
#         global playcount
#         #random colors to assign unique color to each label
#         label_colors = np.random.randint(0,255,size=(len(class_labels),3),dtype='uint8')
#         try:
                
#                 video = cv.VideoCapture(0)
#                 frame_height, frame_width = None, None  #reading video from given path                
#         except:
#                 raise 'Unable to load video'
#         finally:
#                 while True:
#                         frame_grabbed, frames = video.read() #taking each frame from videoz
#                         if not frame_grabbed: #condition to check whether video loaded or not
#                                 break
#                         if frame_width is None or frame_height is None:
#                                 frame_height, frame_width = frames.shape[:2] #detecting object from frame
#                         frames, cls, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels)
#                         data = ""
#                         if len(cls) > 0:
#                                 for i in range(len(cls)):
#                                         data+=cls[i]+","
#                         cv.imshow("Detected Objects", frames)
#                         if len(cls) > 0:
#                                 speak("Detected Objects = "+data, playcount)
#                                 playcount = playcount + 1
#                         if cv.waitKey(5) & 0xFF == ord('q'):
#                                 break    

# if __name__ == '__main__':
#         loadLibraries()
#         deleteDirectory()
#         detectFromVideo()
        


import numpy as np
import cv2 as cv
import subprocess
import time
import os
from ObjectDetection import detectObject, displayImage
import sys
from gtts import gTTS   
from playsound import playsound
from threading import Thread 

global class_labels
global cnn_model
global cnn_layer_names
playcount = 0

def deleteDirectory():
    filelist = [f for f in os.listdir('play') if f.endswith(".mp3")]
    for f in filelist:
        os.remove(os.path.join('play', f))

def speak(data, playcount):
    class PlayThread(Thread):
        def __init__(self, data, playcount):
            Thread.__init__(self) 
            self.data = data
            self.playcount = playcount
        def run(self):
            t1 = gTTS(text=self.data, lang='en', slow=False)
            t1.save(f"play/{self.playcount}.mp3")
            playsound(f"play/{self.playcount}.mp3")
            
    newthread = PlayThread(data, playcount) 
    newthread.start()

def loadLibraries():  # function to load YOLOv3 model weights and class labels
    global class_labels
    global cnn_model
    global cnn_layer_names
    
    class_labels = open('model/yolov3-labels').read().strip().split('\n')
    cnn_model = cv.dnn.readNetFromDarknet('model/yolov3.cfg', 'model/yolov3.weights')
    cnn_layer_names = cnn_model.getLayerNames()
    
    unconnected_layers = cnn_model.getUnconnectedOutLayers()
    
    # Fix indexing issue
    if len(unconnected_layers.shape) == 2:
        unconnected_layers = unconnected_layers.flatten()
    
    cnn_layer_names = [cnn_layer_names[i - 1] for i in unconnected_layers]

def detectFromImage(imagename):
    label_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    try:
        image = cv.imread(imagename)
        image_height, image_width = image.shape[:2]
    except:
        raise Exception('Invalid image path')
    finally:
        image, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, image_height, image_width, image, label_colors, class_labels)
        displayImage(image)

def detectFromVideo():
    global playcount
    label_colors = np.random.randint(0, 255, size=(len(class_labels), 3), dtype='uint8')
    try:
        video = cv.VideoCapture(0)
        frame_height, frame_width = None, None
    except:
        raise Exception('Unable to load video')
    finally:
        while True:
            frame_grabbed, frames = video.read()
            if not frame_grabbed:
                break
            if frame_width is None or frame_height is None:
                frame_height, frame_width = frames.shape[:2]
            frames, cls, _, _, _, _ = detectObject(cnn_model, cnn_layer_names, frame_height, frame_width, frames, label_colors, class_labels)
            data = ",".join(cls)
            cv.imshow("Detected Objects", frames)
            if len(cls) > 0:
                speak(f"Detected Objects = {data}", playcount)
                playcount += 1
            if cv.waitKey(5) & 0xFF == ord('q'):
                break

if __name__ == '__main__':
    loadLibraries()
    deleteDirectory()
    detectFromVideo()
