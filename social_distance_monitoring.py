# Author: Amiya Maity

#collected files
# yolov3.cfg : https://github.com/pjreddie/darknet/blob/master/cfg/yolov3.cfg
# coco.name : https://github.com/pjreddie/darknet/blob/master/data/coco.names
# yolov3.weight : https://pjreddie.com/media/files/yolov3.weights
# video: https://pixabay.com/videos/people-commerce-shop-busy-mall-6387/

# import the necessary packages
import numpy as np
import argparse
import math
import time
import cv2
import os

#Global Variables
PIXEL_THRESH=30


###########################################
#       Social Distance monitoring        #
###########################################
def hconcat_resize_min(im_list, interpolation=cv2.INTER_CUBIC):
    '''
    Input: (list) im_list, (int) interpolation of cv2.
    return: (np.ndarray )
    Description: Concatinating horizontally list of images.
    '''
    h_min = min(im.shape[0] for im in im_list)
    im_list_resize = [cv2.resize(im, (int(im.shape[1] * h_min / im.shape[0]), h_min), interpolation=interpolation)
                      for im in im_list]
    return cv2.hconcat(im_list_resize)

# function to draw bounding box on the detected object with class name
def draw_bounding_box(image,transformed_img,inside_detections,line_between_close_person):
    '''
    Input: (np.ndarray) image, (np.ndarray) transformed_img, (list) inside_detections, (list) line_between_close_person.
    return: (np.ndarray ) image, (np.ndarray ) transformaed_img, (np.ndarray ) final_frame,
    Description: Drawing images line, circle according to cor-ordinate.

    '''
    height,Width,_=transformed_img.shape
    transformed_img=np.zeros((height,Width,3), np.uint8)

    for d in inside_detections:
        x,y,w,h=d['coordinate'][0],d['coordinate'][1],d['coordinate'][2],d['coordinate'][3]
        px,py=d['bird_eye_cord'][0],d['bird_eye_cord'][1]
        col=d['color']

        if col == 'red':
            color = [0, 0, 255]
        else:
            color = [0, 255, 0]

        cv2.rectangle(image, (x, y), (w,h), color, 2)
        cv2.circle(transformed_img, (px,py), 5, color, -1)
        cv2.circle(transformed_img, (px,py), 10, color, 1)


    for l in line_between_close_person:
        transformed_img = cv2.line(transformed_img, l[0], l[1], (0,255,255), 1) 

    # cv2.imshow("Original Frame",image)
    # cv2.imshow("Bird Eye View Frame",transformed_img)
    final_frame = hconcat_resize_min([image, transformed_img])
    
    return image,transformed_img,final_frame #original frame, bird eye view frame,concat frame



def distanceFormula(x1, y1, x2, y2):
    '''
    Input: (int) x1, (int) y1, (int) x2, (int) y2,
    return: (float)
    Description: Calculating distance according to points.
    '''
    f = math.pow((x2 - x1), 2) + math.pow((y2 - y1), 2)
    return math.sqrt(f)

def calculateProspectiveTransformation(image,TOP_LEFT=(50, 50),TOP_RIGHT=(1180, 50),BOTTOM_LEFT=(50, 620),BOTTOM_RIGHT=(1180, 620),PROSPECTIVE_BOX_WIDTH=300,PROSPECTIVE_BOX_HEIGHT=400):
    '''
    Input: (np.ndarray) image, (touple) TOP_LEFT, (touple) TOP_RIGHT, (touple) BOTTOM_LEFT, (touple) BOTTOM_RIGHT, (int) PROSPECTIVE_BOX_WIDTH, (int) PROSPECTIVE_BOX_HEIGHT.
    return: (np.ndarray) matrix3x3, (np.ndarray) transformed_result.
    Description: Calculating prospective transformation and wrap up with image.

    '''
    cv2.circle(image, TOP_LEFT, 5, (0, 255, 255), -1)
    cv2.circle(image, TOP_RIGHT, 5, (0, 255, 255), -1)
    cv2.circle(image, BOTTOM_LEFT, 5, (0, 255, 255), -1)
    cv2.circle(image, BOTTOM_RIGHT, 5, (0, 255, 255), -1)

    image = cv2.line(image, TOP_LEFT, TOP_RIGHT, (255,0,0), 1) 
    image = cv2.line(image, BOTTOM_LEFT, BOTTOM_RIGHT, (255,0,0), 1) 
    image = cv2.line(image,TOP_LEFT, BOTTOM_LEFT, (255,0,0), 1) 
    image = cv2.line(image, TOP_RIGHT, BOTTOM_RIGHT, (255,0,0), 1) 

    pts1 = np.float32([[TOP_LEFT[0],TOP_LEFT[1]], [TOP_RIGHT[0],TOP_RIGHT[1]], [BOTTOM_LEFT[0],BOTTOM_LEFT[1]], [BOTTOM_RIGHT[0],BOTTOM_RIGHT[1]]])
    pts2 = np.float32([[0, 0], [PROSPECTIVE_BOX_WIDTH, 0], [0, PROSPECTIVE_BOX_HEIGHT], [PROSPECTIVE_BOX_WIDTH, PROSPECTIVE_BOX_HEIGHT]])

    matrix3x3 = cv2.getPerspectiveTransform(pts1, pts2)
    transformed_result = cv2.warpPerspective(image, matrix3x3, (PROSPECTIVE_BOX_WIDTH, PROSPECTIVE_BOX_HEIGHT))

    return matrix3x3,transformed_result


def getProspectiveCoordinate(inside_detections,matrix3x3):

    '''
    Input: (list) inside_detections, (np.ndarray) matrix3x3.
    return: (list) inside_detections
    Description: list out original person detected co-ordinate to prospective co-ordinate.

       (x,w) _______
            |       |
            |       |
            |       |
            |___.___|

        basically here prospecting this down dot(.) co-ordinate.
        x=d['coordinate'][0]
        y=d['coordinate'][0]
        w=d['coordinate'][0]  # here w means (x+w)
        h=d['coordinate'][0]  # here h means (y+h)
        because we are getting this value from forward() method line number 227 line code
    '''
    for i,d in enumerate(inside_detections):
        x=int((d['coordinate'][2] + d['coordinate'][0] ) / 2 )
        y=d['coordinate'][3] #for down point 
        a = np.array([[x,y]], dtype='float32')
        a = np.array([a])
        pointsOut = cv2.perspectiveTransform(a, matrix3x3)
        inside_detections[i]['bird_eye_cord']=[pointsOut[0][0][0],pointsOut[0][0][1]]

    return inside_detections

def monitoring(image, detections,ROI,PROSPECTIVE_BOX):
    '''
    Input: (np.ndarray) image, (list) detections, (list) ROI, (list) PROSPECTIVE_BOX.
    return: (np.ndarray) image, (np.ndarray) transformed_img, (np.ndarray)final_frame
    Description: Calculating those people who are in ROI, also permutating close persons and calling others function as well.
    '''

    #ROI    
    TOP_LEFT=ROI[0]
    TOP_RIGHT=ROI[1]
    BOTTOM_LEFT=ROI[2]
    BOTTOM_RIGHT=ROI[3]

    #Prospective shape
    PROSPECTIVE_BOX_WIDTH=PROSPECTIVE_BOX[0]
    PROSPECTIVE_BOX_HEIGHT=PROSPECTIVE_BOX[1]

    matrix3x3,transformed_img=calculateProspectiveTransformation(image,TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT,PROSPECTIVE_BOX_WIDTH,PROSPECTIVE_BOX_HEIGHT)
    
    inside_detections=[]
    line_between_close_person=[]
    for d in detections:
        #checking for detection are inside ROI or not?
        mx=int((d['coordinate'][0] + d['coordinate'][2])/2)
        my=d['coordinate'][3]
        if mx >= TOP_LEFT[0] and mx <= TOP_RIGHT[0] and my >= TOP_LEFT[1] and my <= BOTTOM_LEFT[1]:
            #Inside ROI
            inside_detections.append(d)

    inside_detections=getProspectiveCoordinate(inside_detections,matrix3x3)
    
    length = len(inside_detections)
    for i in range(length):
        for j in range(length):
            if i != j and j > i:
                x1, y1 = inside_detections[i]['bird_eye_cord'][0], inside_detections[i]['bird_eye_cord'][1]
                x2, y2 = inside_detections[j]['bird_eye_cord'][0], inside_detections[j]['bird_eye_cord'][1]
                d = distanceFormula(x1, y1, x2, y2)
                if d < PIXEL_THRESH:
                    inside_detections[i]['color'] = 'red'
                    inside_detections[j]['color'] = 'red'
                    line_between_close_person.append([(x1,y1),(x2,y2)])

    image,transformed_img,final_frame=draw_bounding_box(image,transformed_img,inside_detections,line_between_close_person)
    return image,transformed_img,final_frame #original frame, bird eye view frame,concat frame




#######################################
#      Starting Person Detection      #
#######################################

def initializeNet(args):
    '''
    Input: ( Namespace object ) args
    return: (dnn net) initialized network, (list) labels 
    Description: Initialize the yolo network and reading coco names file.
    '''
    cfgFile=args.config
    weightFile=args.weights
    print("[INFO] loading YOLO from disk...")
    net = cv2.dnn.readNetFromDarknet(cfgFile, weightFile)
    labels=open(args.classes).read().strip().split("\n")
    return net,labels

def inputblobToNetwork(image, net):
    '''
    Input: ( np.ndarray ) image, (dnn net ) net.
    return: (dnn net ) net, (np.ndarray) image, (int) width, (int) height
    Description: Input image to dnn network with proper size. i.e; 416,320 etc .
    '''
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392
    # create input blob
    blob = cv2.dnn.blobFromImage(
        image, scale, (416, 416), (0, 0, 0), True, crop=False)
    # set input blob for the network
    net.setInput(blob)
    return net, image,Width, Height


# function to get the output layer names
# in the architecture
def get_output_layers(net):
    '''
    Input: (dnn net ) net.
    return: (list) output layers.
    Description: According to network, return output layers.
    '''
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i[0] - 1]
                     for i in net.getUnconnectedOutLayers()]
    return output_layers


def forward(net, Width, Height):
    '''
    Input: (dnn net) net, (int) Width, (int) Height.
    return: (list) indices,(list) class_ids,(list) confidences,(list) boxes.
    Description: Run inference through the network and predict object and listing out indices, class_id,confidences, and bounding box.
    '''
    # run inference through the network
    # and gather predictions from output layers
    outs = net.forward(get_output_layers(net))
    # initialization
    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4

    # for each detetion from each output layer
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    # apply non-max suppression
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    return indices, class_ids, confidences, boxes

def getDetections(indices, classes, class_ids, confidences, boxes):
    '''
    Input: (list) indices,(list) class_ids,(list) confidences,(list) boxes.
    Output: (list) detections_list.
    Description: According to indices listing detections with dictionary data.
    '''
    # go through the detections remaining
    # after nms and draw bounding box
    detections_list = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]

        if str(classes[class_ids[i]]) == 'person':
            detections = dict()
            detections['label'] = str(classes[class_ids[i]])
            detections['confidances'] = confidences[i]
            detections['color'] = 'green'
            detections['coordinate'] = [round(x), round(y), round(x+w), round(y+h)]
            detections_list.append(detections)

    return detections_list


def monitoring_from_video(args,net,fileReadclasses):
    '''
    Input: (Namespace object) args, (dnn net) net, (list) fileReadclasses.
    Return: None.
    Description: Framing input video and calling others functions to get final frame.
    '''

    # define a video capture object 
    # vid = cv2.VideoCapture(0) 
    vid = cv2.VideoCapture(args.input) 
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID') 

    output_path=args.input
    output_path=output_path.split('.')[0]+"_drawn."+output_path.split('.')[1]
    print(output_path)
    writer = cv2.VideoWriter(output_path, fourcc, 30.0, (1280, 720)) 

    ret, frame = vid.read() 
    height,Width,_=frame.shape 
    print("Original Frame Width: {} Height: {}\n\n".format(Width,height))
    #ROI  Selection   
    TOP_LEFT=(50, 50)
    TOP_RIGHT=(1180, 50)
    BOTTOM_LEFT=(50, 620)
    BOTTOM_RIGHT=(1180, 620)

    #Prospective shape
    PROSPECTIVE_BOX_WIDTH=300
    PROSPECTIVE_BOX_HEIGHT=400

    #Bind to list
    ROI=[TOP_LEFT,TOP_RIGHT,BOTTOM_LEFT,BOTTOM_RIGHT]
    PROSPECTIVE_BOX=[PROSPECTIVE_BOX_WIDTH,PROSPECTIVE_BOX_HEIGHT]

    
    while(vid.isOpened()): 
        # Capture the video frame 
        # by frame 
        ret, frame = vid.read() 
        
        net, image,imgWidth, imgHeight = inputblobToNetwork(frame, net)
        indices, class_ids, confidences, boxes = forward(net, imgWidth, imgHeight)        
        detections_list = getDetections(indices, fileReadclasses, class_ids, confidences, boxes)
        #original frame, bird eye view frame,concat frame
        image,bird_eye_view,concat_frame = monitoring(image, detections_list,ROI,PROSPECTIVE_BOX)
        concat_frame = cv2.resize(concat_frame, (1280, 720),interpolation = cv2.INTER_NEAREST) 

        height,width,_=concat_frame.shape
        print(height,width)
        writer.write(concat_frame)
        cv2.imshow("Final Frame: ",concat_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'): 
            break
    # After the loop release the cap object 
    vid.release() 
    # global writer
    writer.release()
    # Destroy all the windows 
    cv2.destroyAllWindows() 


def main(args):
    '''
    Input: (Namespace object) args.
    return: None.
    Description: calling function.

    '''
    net, fileReadclasses = initializeNet(args)
    monitoring_from_video(args,net,fileReadclasses)
    


if __name__ == '__main__':
    # handle command line arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-i', '--input', required=True, help='path to input video')
    ap.add_argument('-c', '--config', required=True,help='path to yolo config file')
    ap.add_argument('-w', '--weights', required=True,help='path to yolo pre-trained weights')
    ap.add_argument('-cl', '--classes', required=True,help='path to text file containing class names')
    args = ap.parse_args()
    main(args)
    
