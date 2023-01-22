import os
import numpy as np
import cv2
import glob  # to read files
from tqdm import tqdm # to check the progress
import math
import json

# This class performes rotation of images, as well the finds the bounding box coordinates of rotated object
# Defining a class to perform Augmentation.
# Here ony Rotating the images is considered
# Perform YOLO Rotation
class yoloRotate:
    def __init__(self, filename,image_ext,angle):
        """
        1. Checking the image paths
        2. Initializing the Images
        3. Defining the transformation Matrix
        """
        assert os.path.isfile(filename + image_ext)
        assert os.path.isfile(filename + '.json')
        
        self.filename = filename
        self.image_ext = image_ext
        self.file_ext = '.json'
        self.angle = angle
        
        # read image using cv2
        self.image = cv2.imread(self.filename + self.image_ext,1)


    def cv_to_yolo(self, main_bbox, ht, wd):
        """
        This function works similar to the above xml_to_yolo(). The only point of difference is the bbox size. 
        main_bbox = The BB coords along with the index is passed. This is the output of the rotated images.
        """
        bb_width = main_bbox[3] - main_bbox[1]
        bb_height = main_bbox[4] - main_bbox[2]
        bb_cx = (main_bbox[1] + main_bbox[3])/2
        bb_cy = (main_bbox[2] + main_bbox[4])/2

        return main_bbox[0], round(bb_cx/wd, 6), round(bb_cy/ht, 6), round(bb_width/wd, 6), round(bb_height/ht, 6)    


    def yoloFormattocv(self, x1, y1, x2, y2, H, W):
        bbox_width = x2 * W
        bbox_height = y2 * H
        center_x = x1 * W
        center_y = y1 * H
        voc = []
        voc.append(center_x - (bbox_width/2))
        voc.append(center_y - (bbox_height/2))
        voc.append(center_x + (bbox_width/2))
        voc.append(center_y + (bbox_height/2))
        return [int(v) for v in voc]


    def translatePoint(self, point):
        # Compute new coordinate of each point by polar coordinate transformation

        theta = self.angle * math.pi/180
        H, W = self.image.shape[:2]
        cy = H/2
        cx = W/2

        # Convert cartesian coordinate to polar coordinate
        new_w = int(W * abs(math.sin(theta)) + H * abs(math.cos(theta)))
        new_h = int(W * abs(math.cos(theta)) + H * abs(math.sin(theta)))

        cx_n = new_w/2
        cy_n = new_h/2

        p2 = [cx, cy]
        r = math.dist(point, p2)

        dx = point[0] - p2[0]
        dy = point[1] - p2[1]

        # Check for division by zero
        if dx == 0 :
            if dy > 0 :
                phi = math.pi * 3/2
            elif dy < 0:
                phi  = math.pi/2
        else :
            phi = math.atan(dy/-dx)

        if dy == 0 :
            if dx < 0 :
                phi = math.pi
            elif dx >= 0:
                phi = 0

        if phi < 0 or dy > 0:
            phi += math.pi

        alpha = theta + phi
        dx_n = int(r * math.cos(alpha))# relative to cx , cy
        dy_n = int(r * math.sin(alpha))
        x_n = cx_n - dy_n
        y_n = cy_n + dx_n

        return [y_n, x_n]

    def rotateYoloBB(self):
        # Opening JSON file
        myFile = open(self.filename + self.file_ext,"tr")

        # returns JSON object as 
        # a dictionary
        data = json.load(myFile)
        label_name_to_value = {"label": 0}

        # Iterating through the json
        # list


        polygons = []
        labelValues = []

        for row in sorted(data['shapes'], key=lambda x: x["label"]):
            polygon = []
            label_name = row["label"]
            if label_name in label_name_to_value:
                labelValues.append(label_name_to_value[label_name])
            else:
                labelValues.append(-1)

            for r in  row['points']:
                polygon.append(r)

            polygons.append(polygon)
        
        # Closing file
        myFile.close()

        new_bbox = []
        for polygon, lv in zip(polygons, labelValues):
            points = []
            for p in polygon:
                points.append(p)

                newPoints = []
                xPoints = []
                yPoints = []

                for point in points:
                    translatedPoint = self.translatePoint(point)
                    newPoints.append(translatedPoint)
                    xPoints.append(translatedPoint[0])
                    yPoints.append(translatedPoint[1])

            minX = np.min(xPoints)
            minY = np.min(yPoints)
            maxX = np.max(xPoints)
            maxY = np.max(yPoints)
            new_bbox.append([lv, minX, minY, maxX, maxY])

        return new_bbox


    def rotate_image(self):
        """
        This function focuses on rotating the image to a particular angle. 
        """
        height, width = self.image.shape[:2]
        img_c = (width/2, height/2) # Image Center Coordinates
        rotation_matrix = cv2.getRotationMatrix2D(img_c, self.angle, 1.) # Rotating Image along the actual center
        
        abs_cos = abs(rotation_matrix[0,0])  # Cos(angle)
        abs_sin = abs(rotation_matrix[0,1])  # sin(angle)
        
        # New Width and Height of Image after rotation
        bound_w = int(height * abs_sin + width * abs_cos)
        bound_h = int(height * abs_cos + width * abs_sin)
        
        # subtract the old image center and add the new center coordinates
        rotation_matrix[0,2]+= bound_w/2 - img_c[0]
        rotation_matrix[1,2]+= bound_h/2 - img_c[1]
        
        # rotating image with transformed matrix and new center coordinates
        rotated_matrix = cv2.warpAffine(self.image, rotation_matrix,(bound_w, bound_h))
        
        return rotated_matrix

if __name__=="__main__":   
    input_dir = "dataset/"
    output_dir = "output/"
    
    # create the labels folder (output directory)
    isExist = os.path.exists(output_dir)

    if not isExist:
        os.mkdir(output_dir)
    
    ext = ["jpg","png"]

    for e in tqdm(ext):
        files = glob.glob(os.path.join(input_dir, '*.'+e))
        for filename in files:
            file = filename.split(".")
            
            image_name = file[0]
            image_ext = "."+file[1]            

            # Rotated Image Augmentation        
            angles = [45, 90, 180, 225, 270]
            for angle in angles:
                im = yoloRotate(image_name,image_ext,angle)
                bbox = im.rotateYoloBB()  # new BBox values, after rotation
                rotated_image= im.rotate_image()  # rotated_image
                
                # writing into new folder
                f = image_name
                file = f.split("/")

                file_name= file[-1]+'_'+str(angle) + '.txt'
                #saving the Bbox ccoordinates in YOLO format in same folder as its augmented image data
                for bb in bbox:
                    #cv2.rectangle(rotated_image,(int(bb[1]), int(bb[2])), (int(bb[3]), int(bb[4])), [255, 0, 255], 5)
                    with open(os.path.join(output_dir,file_name),'a') as fout:
                        fout.writelines(
                        ' '.join(map(str, im.cv_to_yolo(bb, rotated_image.shape[0], rotated_image.shape[1]))) + '\n') 
                    cv2.imwrite(os.path.join(output_dir + file[-1] + '_' + str(angle) + image_ext), rotated_image)
                    
                
                        
