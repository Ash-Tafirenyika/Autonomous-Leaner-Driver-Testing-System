import cv2
import numpy as np


#Detects objects in videos
class Object_dection():
    
    def __init__(self, imgs):
        
        self.imgs = imgs
    
    def Process(self):
        
        #cnn network setup
        net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
        classes = []
        #extracts labels and populates them in classes
        with open("coco.names", "r") as f:
            classes = [line.strip() for line in f.readlines()]
        #get layers from cnn     
        layer_names = net.getLayerNames()
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        
        #generates random colors to be applied to object detected
        colors = np.random.uniform(0, 255, size=(len(classes), 3))
        self.imgs = cv2.resize(self.imgs, None, fx=0.4, fy=0.4)
        height, width, channels = self.imgs.shape
    
        # Detecting objects
        blob = cv2.dnn.blobFromImage(self.imgs, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
        
        net.setInput(blob)
        outs = net.forward(output_layers)
        
        # Showing informations on the screen
        class_ids = []
        confidences = []
        boxes = []
        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5:
                    # Object detected
                    center_x = int(detection[0] * width)
                    center_y = int(detection[1] * height)
                    w = int(detection[2] * width)
                    h = int(detection[3] * height)
        
                    # Rectangle coordinates
                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)
        
                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)
        
        indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        print(indexes)
        font = cv2.FONT_HERSHEY_COMPLEX
        for i in range(len(boxes)):
            if i in indexes:
                x, y, w, h = boxes[i]
                label = str(classes[class_ids[i]])
                color = colors[class_ids[i]]
                cv2.rectangle(self.imgs, (x, y), (x + w, y + h), color, 6)
                cv2.putText(self.imgs, label, (x + 4, y - 18), font, 1, color, 3)
        return self.imgs
                