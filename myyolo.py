from scipy.spatial import distance as dist
import numpy as np
import argparse
import imutils
import cv2
import os
import math
import logging


class MainWindow:
    def __init__(self, width=50, height=300):
        self.width = width
        self.height = height
        self.FR = np.zeros((self.height, self.width, 3), np.uint8)
        self.color = (255, 255, 255)
        self.FR[:] = self.color
        self.infoWidth = 150
        self.infoHeght = self.width

    def getFrame(self):
        while True:
            cv2.line(self.FR, (10, 20), (140, 20), (0, 0, 0), 2)
            # TITLE
            cv2.putText(self.FR, "COVID 19 ", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            cv2.putText(self.FR, "Distanciamiento Social", (0, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # Bounding Box
            cv2.rectangle(self.FR, (2, 100), (148, 200), (100, 100, 100), 2)
            cv2.putText(self.FR, " Las cajas muestran ", (2, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 0), 1)
            cv2.putText(self.FR, " el nivel de contagio.", (2, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 100, 0), 1)
            cv2.putText(self.FR, "-Rojo:Peligro de Contagio", (4, 160),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            cv2.putText(self.FR, "-Verde:Bajo Riesgo ", (4, 180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            # Total frame  info
            totalPerson = "Total Personas: 10"
            totalLowRisk = "Total Bajo Riesgo: 4"
            totalHighRisk = "Total Alto Riesgo: 6"

            # Rectangle fow show frame info
            cv2.rectangle(self.FR, (2, 220), (148, 400), (100, 100, 100), 2)
            cv2.putText(self.FR, "FrameInfo:", (20, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 0), 1)

            cv2.putText(self.FR, totalPerson, (2, 260),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(self.FR, totalHighRisk, (2, 280),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
            cv2.putText(self.FR, totalLowRisk, (2, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)

            cv2.imshow('Social distancing analyser', self.FR)
            cv2.waitKey(1)
# info=MainWindow(width=500,height=600)
# info.getFrame()


class ObjectDetector:
    def __init__(self, inputPath="", outputPath=""):
        # load the COCO class labels our YOLO model was trained on
        self.labelsPath = os.path.sep.join(["yolo-coco", "coco.names"])
        self.LABELS = open(self.labelsPath).read().strip().split("\n")

        # initialize a list of colors to represent each possible class label
        np.random.seed(42)
        self.COLORS = np.random.randint(0, 255, size=(len(self.LABELS), 3),
                                        dtype="uint8")
        # derive the paths to the YOLO weights and model configuration
        self.weightsPath = os.path.sep.join(["yolo-coco", "yolov3.weights"])
        self.configPath = os.path.sep.join(["yolo-coco", "yolov3.cfg"])

        # load our YOLO object detector trained on COCO dataset (80 classes)
        print("[INFO] loading YOLO from disk...")
        self.net = cv2.dnn.readNetFromDarknet(
            self.configPath, self.weightsPath)
        # determine only the *output* layer names that we need from YOLO
        self.ln = self.net.getLayerNames()
        self.ln = [self.ln[i[0] - 1]
                   for i in self.net.getUnconnectedOutLayers()]

        # initialize the width and height of the frames in the video file
        self.W = None
        self.H = None

        # initialize the video stream and pointer to output video file, then
        # start the FPS timer
        print("[INFO] accessing video stream...")
        self.vs = cv2.VideoCapture(inputPath+"6seg_acelerado.mp4")
        self.writer = None
        self.F = 615
        self.pos = {}
        self.coordinates = {}
        self.confidence={}
    def objectDetector(self):
        while True:
            (grabbed, frame) = self.vs.read()
            if not grabbed:
                break
            frame = imutils.resize(frame, width=600)
            results = self.detectPerson(
                frame, personIdx=self.LABELS.index("person"))

    def createInfoFrame(self, width, height, low, height1):
        FR = np.zeros((height, width+150, 3), np.uint8)
        color = (255, 255, 255)
        FR[:] = color
        cv2.line(FR, (10, 20), (140, 20), (0, 0, 0), 2)
        # TITLE
        cv2.putText(FR, "COVID 19 ", (30, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        cv2.putText(FR, "Distanciamiento Social", (0, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        # Bounding Box
        cv2.rectangle(FR, (1, 100), (148, 200), (100, 100, 100), 2)
        cv2.putText(FR, " Los rectAngulos muestran ", (2, 120),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 0), 1)
        cv2.putText(FR, " el riesgo de contagio.", (2, 140),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (100, 100, 0), 1)
        cv2.putText(FR, "-Rojo: Alto Riesgo", (4, 160),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 0, 255), 1)
        cv2.putText(FR, "-Verde: Bajo Riesgo ", (4, 180),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.34, (0, 255, 0), 1)
        # Total frame  info
        totalPerson = "Total Personas:"+str(low+height1)
        totalLowRisk = "Total Bajo Riesgo:"+str(low)
        totalHighRisk = "Total Alto Riesgo:"+str(height1)

        # Rectangle fow show frame info
        cv2.rectangle(FR, (2, 220), (148, 400), (100, 100, 100), 2)
        cv2.putText(FR, "FrameInfo:", (20, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 100, 0), 1)

        cv2.putText(FR, totalPerson, (2, 260),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 1)
        cv2.putText(FR, totalHighRisk, (2, 280),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
        cv2.putText(FR, totalLowRisk, (2, 300),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        return FR

    def detectPerson(self, frame, personIdx):
        (H, W) = frame.shape[:2]
        results = []
        blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416),
                                     swapRB=True, crop=False)
        self.net.setInput(blob)
        layerOutputs = self.net.forward(self.ln)
        boxes = []
        centroids = []
        confidences = []
        for output in layerOutputs:
            for detection in output:
                scores = detection[5:]
                classID = np.argmax(scores)
                confidence = scores[classID]
                if classID == personIdx and confidence > 0.5:
                    box = detection[0:4] * np.array([W, H, W, H])
                    (centerX, centerY, width, height) = box.astype("int")
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    boxes.append([x, y, int(width), int(height)])
                    centroids.append((centerX, centerY))
                    confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
        if len(idxs) > 0:
            for i in idxs.flatten():
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                r = (confidences[i], (x, y, x + w, y + h), centroids[i])
                results.append(r)
        for i in range(0, len(results)):
            (confidence, box, centroid) = results[i]
            (startX, startY, endX, endY) = box
            # print("startx",startX,"starty",startY,"endx",endX,"endy",endY)
            xc, yc = centroid
            self.coordinates[i] = (startX, startY, endX, endY)
            self.confidence[i]=confidence
            midOfX = round((startX+endX)/2, 4)
            midOfY = round((startY+endY)/2, 4)
            ht = round(endY-startY, 4)
            distance = (self.F * 165)/ht
            midOfX_cm = (midOfX * distance) / self.F
            midOfY_cm = (midOfY * distance) / self.F
            self.pos[i] = (midOfX_cm, midOfY_cm, distance)
           
            #cv2.rectangle(frame, (startX, startY), (endX, endY), (255,0,0), 2)
            #cv2.circle(frame, (int(midOfX_cm), int(midOfY_cm)),5, (0, 255, 0), 1)
            # cv2_imshow(frame)

        proximity = []
        # for i in self.pos.keys():
        #     for j in self.pos.keys():

        #         if i < j:
        #             dist = math.sqrt(pow(self.pos[i][0] - self.pos[j][0], 2) + pow(self.pos[i][1] - self.pos[j][1], 2) + pow(self.pos[i][2] - self.pos[j][2], 2))
        #             print(dist)
        #         # Checking threshold distance - 175 cm
        #             if dist < 175:
        #                 proximity.append(i)
        #                 proximity.append(j)
        #                 warning_label = "Maintain Safe Distance. Move away!"
        #                 cv2.putText(frame, warning_label, (50, 50),
        #                             cv2.FONT_HERSHEY_DUPLEX, 0.5, (0, 255, 0), 1)

        ##############################################################################################
        for i in range(0, len(results)):
            # (confidence, box, centroid) = results[i]
            for j in range(i+1, len(results)):

                (confidence, boxi, centroid) = results[i]
                (startXi, startYi, endXi, endYi) = boxi

                (confidence, boxj, centroid) = results[j]
                (startXj, startYj, endXj, endYj) = boxj

                hbbf = endYi - startYi  # height bounding box frente / atras
                hbba = endYj - startYj
               # height = endYi - startYi

                if endYj > endYi:
                    hbbf = endYj - startYj
                    hbba = endYi - startYi
                    #height = endYj - startYj
                print(i, j, hbbf, hbba, abs(endYi-endYj), hbbf)

                if hbba > (hbbf-hbba) and abs(endYi-endYj) <= hbbf*(5.0/10) and abs(endXi-endXj) <= hbbf*(5.0/10):
                    proximity.append(i)
                    proximity.append(j)
##############################################################################################

        low = 0
        height = 0
        for i in range(0,len(results)):
            if i in proximity:
                color = [0, 0, 255]
                height += 1
            else:
                color = [0, 255, 0]
                low += 1
            (x, y, w, h) = self.coordinates[i]
            FR = self.createInfoFrame(
                frame.shape[1], frame.shape[0], low, height)
            cv2.rectangle(frame, (x, y), (w, h), color, 2)
            label="person:"+str(round(self.confidence[i],1)*100)
            cv2.putText(frame,label,(x,y),cv2.FONT_HERSHEY_SIMPLEX,0.25,(0,255,0),1)
        FR[:, 150:] = frame
        frame = FR
       
        if 1 > 0:
            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            # if the `q` key was pressed, break from the loop
            # if key == ord("q"):
            #     break
        frame = frame[:, 150:]

    def computeDistance(self):
        pass

    def infoFrame():
        pass

    def open_video_stream(self, filename, return_props=False):
        log = logging.getLogger('test_video')
        try:
            with open(filename, 'rb') as f:
                pass
        except IOError as e:
            raise RuntimeError(e)

        cap = cv2.VideoCapture(filename)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_rate = cap.get(cv2.CAP_PROP_FPS)
        if not cap.isOpened():
            raise RuntimeError(
                "Failed to open video file. Is ffmpeg properly configured?")
        log.info("Loading {} [{} frames]...".format(filename, num_frames))

        if return_props:
            return cap, {'num_frames': num_frames, 'frame_rate': frame_rate}
        else:
            return cap

    def getVideoInfo(self, videoPath=""):
        video_stream, props = self.open_video_stream(
            videoPath, return_props=True)
        print("{}: {} fps, {} frames".format(
            videoPath, props['frame_rate'], props['num_frames']))


objDetector = ObjectDetector()
objDetector.objectDetector()
# objDetector.getVideoInfo("pedestrians.mp4")
