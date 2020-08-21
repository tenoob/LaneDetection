import cv2
import numpy as np
import matplotlib.pyplot as plt



def canny(image):
    # step 2 convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # step 3 apply gaussian blur to reduce noise and smooth image
    blur = cv2.GaussianBlur(gray, (5, 5), 0)  # optional

    # step 4 Canny to find edges
    canny = cv2.Canny(blur, 50, 150)
    return canny

def region_of_interest(image):
    # step 5 Region of Interest
    height=image.shape[0]
    #the polygons are found by using the matplotlib canny_img
    polygons = np.array([[(200 , height) , (1100 ,height),(550,250)]] , dtype=np.int32)
    mask=np.zeros_like(image)
    cv2.fillPoly(mask,polygons,255)

    # step 6 Bitwise And
    masked_image=cv2.bitwise_and(image,mask)
    return masked_image

def display_line(image,lines):
    line_image=np.zeros_like(image)
    if lines is not None:
        for line in lines:
            x1,y1,x2,y2 = line.reshape(4)
            #.reshape can be removed as we have converted the image into 1 dimensional
            cv2.line(line_image,(x1,y1),(x2,y2),(255,0,0),10)
    return line_image

def average_slope_intersept(image,lines):
    #step 9(OPTIONAL) OPTIMIZATION
    left_fit=[]
    right_fit=[]
    for line in lines:
        x1,y1,x2,y2 = line.reshape(4)
        parameters=np.polyfit((x1,x2),(y1,y2),1)
        slope=parameters[0]
        intersept=parameters[1]
        if slope<0:
            left_fit.append((slope,intersept))
        else:
            right_fit.append((slope,intersept))
    left_fit_avg = np.average(left_fit,axis=0)
    right_fit_avg = np.average(right_fit,axis=0)
    left_line=make_coordinates(image,left_fit_avg)
    right_line=make_coordinates(image,right_fit_avg)
    return np.array([left_line,right_line])


def make_coordinates(image,line_parameters):
    slope,intersept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1*(3/5))
    x1 = int((y1-intersept)/slope)
    x2 = int((y2-intersept)/slope)
    return np.array([x1,y1,x2,y2])

#step 1 load video
cap = cv2.VideoCapture("Resources/test2.mp4")
while(cap.isOpened()):
    _,frame = cap.read()
    canny_img = canny(frame)

    croped_image = region_of_interest(canny_img)

    # step 7 Hough Transform
    lines = cv2.HoughLinesP(croped_image, 2, np.pi / 100, 100, np.array([]), minLineLength=40, maxLineGap=5)
    averaged_lines = average_slope_intersept(frame, lines)

    # step 8 combine line and image
    line_image = display_line(frame, averaged_lines)  # the seecond parameter was  lines insted of averaged_lines
    final_image = cv2.addWeighted(frame, 0.8, line_image, 1, 1)

    # plt.imshow(canny_img)
    cv2.imshow("Final", final_image)
    if cv2.waitKey(1) & 0xff==ord('q'):
        break
cap.release()
cv2.destroyAllWindows()