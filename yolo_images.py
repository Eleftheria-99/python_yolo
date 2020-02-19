#!/usr/bin/python
import sys
import cv2
import numpy as np    # NumPy's arrays are not flexible like Python lists, you can store only same data type in each column.


#directory of the yolo model
with open("yolo/coco.names", 'r') as f:
    classes = [line.strip() for line in f.readlines()]   #The strip() method returns a copy of the string with both leading and trailing characters removed.
net = cv2.dnn.readNet("yolo/yolov3.weights", "yolo/yolov3.cfg")    #?

# function to get the output layer names in the architecture
def get_output_layers(net):    
    layer_names = net.getLayerNames()    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    return output_layers

def draw_prediction(img, class_id, confidence, x, y, x_plus_w, y_plus_h,classes,COLORS):
    label = str(classes[class_id])
    color = COLORS[class_id]
    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)
    cv2.putText(img, label, (x-10,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

def read_frame(file):      #load and return picture from the selected file 
    image = cv2.imread(file)
    return image

def detect(classes,net,image,class_of_interest):     #image= the image that method read_frame returned  
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392    
    
    # generate different colors for different classes
    COLORS = np.random.uniform(0, 255, size=(len(classes), 3)) #Samples are uniformly distributed over the half-open interval [low, high) (includes low, but excludes high). In other words, any value within the given interval is equally likely to be drawn by uniform.
    #classes = everything from the file we opened above 
    
    # create input blob
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)#[blobFromImage] creates 4-dimensional blob from image. Optionally resizes and crops image from center, subtract mean values, scales values by scalefactor, swap Blue and Red channels.
    #(image we want to preprocess before passing it through our deep neural network for classification, we can optionally scale our images by some factor,spatial size that the Convolutional Neural Network expects, mean subtraction values, OpenCV assumes images are in BGR channel order; however, the `mean` value assumes we are using RGB order. To resolve this discrepancy we can swap the R and B channels in image  by setting this value to `True` 
   
    net.setInput(blob)  #Sets the new value for the layer output blob.
    
    # run inference through the network and gather predictions from output layers  #?
    outs = net.forward(get_output_layers(net))
    
    #initialization 
    class_ids = []
    confidences = []
    boxes = []
    
    #confidence of detecet objects 0.1 - 1 -> 1% - 100%
    conf_threshold = 0.7
    nms_threshold = 0.4
    
    # for each detetion from each output layer 
    for out in outs:
        for detection in out:
            scores = detection[5:]  #5 till the end 
            class_id = np.argmax(scores)  #Returns the indices of the maximum values along an axis.
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
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    results = []
    
    # go through the detections remaining after nms and draw bounding box
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        draw_prediction(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h),classes,COLORS)
        results.append({'class':classes[class_ids[i]] , 'confidence':confidences[i]})
    
    number_of(class_of_interest,results)   #?
    preview(image)
    save_file(image)

def number_of(for_detection,results):
    counter = 0
    for i in results:
        if i['class'] == for_detection:
            counter+=1
    print(('Detected {} items of class {}.').format(counter,for_detection))    
        
def preview(image):
    cv2.imshow("object detection", image) #cv2. imshow() method is used to display an image in a window. The window automatically fits to the image size. 
    cv2.waitKey()    #You can create as many windows as you wish, but with different window names. cv2. waitKey() is a keyboard binding function. ... The function waits for specified milliseconds for any keyboard event.

def save_file(image):
    cv2.imwrite("object-detection.jpg", image) #cv2. imwrite() method is used to save an image to any storage device. This will save the image according to the specified format in current working directory.
    cv2.destroyAllWindows()

    
if __name__ == "__main__":   #main
    #default directory of the image
    directory_of_image = "images/cl2.jpg"
    #default class, read yolo/coco.names to select a available class 
    class_of_interest = "person"  #?

    #check for arguments
    if len(sys.argv) > 2:
        directory_of_image = sys.argv[1]
        class_of_interest = sys.argv[2]
    elif len(sys.argv) > 1:
        directory_of_image = sys.argv[1]
    
    frame = read_frame(directory_of_image)           #load and return picture from the selected file 
    detect(classes,net,frame,class_of_interest)
    
    
    
        
    
