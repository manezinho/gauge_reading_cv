'''  
Copyright (c) 2017 Intel Corporation.
Licensed under the MIT license. See LICENSE file in the project root for full license information.
'''

import cv2
import numpy as np
#import paho.mqtt.client as mqtt
import time
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd


def avg_circles(circles, b):
    avg_x=0
    avg_y=0
    avg_r=0
    for i in range(b):
        #optional - average for multiple circles (can happen when a gauge is at a slight angle)
        avg_x = avg_x + circles[0][i][0]
        avg_y = avg_y + circles[0][i][1]
        avg_r = avg_r + circles[0][i][2]
    avg_x = int(avg_x/(b))
    avg_y = int(avg_y/(b))
    avg_r = int(avg_r/(b))
    return avg_x, avg_y, avg_r

# find distance between two points
def dist_2_pts(x1, y1, x2, y2):
    #print np.sqrt((x2-x1)^2+(y2-y1)^2)
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# calibrate image
def calibrate_gauge(img, gauge_number, file_type):
    '''
        This function should be run using a test image in order to calibrate the range available to the dial as well as the
        units.  It works by first finding the center point and radius of the gauge.  Then it draws lines at hard coded intervals
        (separation) in degrees.  It then prompts the user to enter position in degrees of the lowest possible value of the gauge,
        as well as the starting value (which is probably zero in most cases but it won't assume that).  It will then ask for the
        position in degrees of the largest possible value of the gauge. Finally, it will ask for the units.  This assumes that
        the gauge is linear (as most probably are).
        It will return the min value with angle in degrees (as a tuple), the max value with angle in degrees (as a tuple),
        and the units (as a string).
    '''
    height, width = img.shape[:2]
    # grayscale it
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    # blur image
    img_blur = cv2.medianBlur(gray,5)
    thresh = 175
    maxvalue = 255
    th_gaussian = cv2.adaptiveThreshold(gray,maxvalue,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
    edges = cv2.Canny(img,100,200)


    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray,(5,5),0)
    ret_otsu_blur,th_otsu_blur = cv2.threshold(blur,0,maxvalue,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_otsu,th_otsu = cv2.threshold(gray,0,maxvalue,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # min and max size of circles
    r_min = np.int(height*0.30)
    r_max = np.int(height*0.45)

    # param1 controls sensitivity; how strong the edges need to be
    # param2 will set how many points it needs to declare that it's found a circle. "ideal value of param 2 will be related to the circumference of the circles"
    param1_default = 200
    param2_default = 10
    param1 = 300
    param2 = 30

    circles = cv2.HoughCircles(blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=300, param2=30, minRadius=r_min, maxRadius=r_max)

    # average found circles, found it to be more accurate than trying to tune HoughCircles parameters to get just the right one
    a, b, c = circles.shape
    circles = circles[0] # the first index has ALL circles
    print('{} total number of circles have been found before filtering'.format(circles.shape[0]))
    # # calculate center x,y of image
    picture_center_x = np.around(width/2)
    picture_center_y = np.around(height/2)

    # calculate each circle's distance from picture center
    circles = pd.DataFrame(circles)
    circles.columns = ['x','y','r']
    circles['dist from picture center'] = np.sqrt((circles['x'] - picture_center_x)**2 + (circles['y'] - picture_center_y)**2 )
    # calculate standard deviation
    circle_std = circles['dist from picture center'].std() 
    # print
    print('standard deviation of circle center and picture center distance: {}'.format(circle_std))
    print(circles)

    print('height and width are:')
    print(height,width)

    ########################################
    ##### control hyperparameters
    ########################################

    # check if there are more than 50 circles
    # if circles.shape[0] > 50:
    #     while circles.shape[0] > 50:
    #         # reduce the number of circles by recalculating circles
    #         # increase param 1
    #         param1 = param1 + 50
    #         circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=param1, param2=param2, minRadius=r_min, maxRadius=r_max)
    # if circles_std <



    # circles = cv2.HoughCircles(th_otsu_blur, cv2.HOUGH_GRADIENT, 1, img.shape[0]/64, param1=300, param2=25, minRadius=r_min, maxRadius=r_max)

    # printing all the circles on image to see what they look like

    # draw center and circle
    # Draw a circle with blue line borders of thickness of 3 px 
    # image = cv2.circle(image, center_coordinates, radius, color, thickness) 

    # circles = circles.apply(lambda row: np.int(np.around(row)))
    # print(circles)

    # circles.apply(lambda row: cv2.circle(img, (row['x'], row['y']), row['r'], (255, 0, 0), 1, cv2.LINE_AA), axis = 1)
    # circles.apply(lambda row: cv2.circle(img, (row['x'], row['y']), 2, (0, 0, 255), 1, cv2.LINE_AA), axis = 1)
    # cv2.imwrite('gauge-{}-circles.{}'.format(gauge_number, file_type), img)

    # circles.apply(lambda row: print(row))

    for index,row in circles.iterrows():
        x = int(np.around(row['x']))
        y = int(np.around(row['y']))
        r = int(np.around(row['r']))
        
        # draw outer circle
        cv2.circle(img, (x, y), r, (255, 0, 0), 1, cv2.LINE_AA)
        # this r = 2 is arbitrary center circle size (center of circle)
        cv2.circle(img, (x, y), 2, (0, 0, 255), 1, cv2.LINE_AA)  
        cv2.imwrite('gauge-{}-circles.{}'.format(gauge_number, file_type), img)

    x = 449
    y = 390
    r = 309

    #for testing, output circles on image
    #cv2.imwrite('gauge-{}-circles.{}'.format(gauge_number, file_type), img)

    #for calibration, plot lines from center going out at every 10 degrees and add marker
    #for i from 0 to 36 (every 10 deg)

    '''
    goes through the motion of a circle and sets x and y values based on the set separation spacing.  Also adds text to each
    line.  These lines and text labels serve as the reference point for the user to enter
    NOTE: by default this approach sets 0/360 to be the +x axis (if the image has a cartesian grid in the middle), the addition
    (i+9) in the text offset rotates the labels by 90 degrees so 0/360 is at the bottom (-y in cartesian).  So this assumes the
    gauge is aligned in the image, but it can be adjusted by changing the value of 9 to something else.
    '''
    separation = 10.0 #in degrees
    interval = int(360 / separation)
    p1 = np.zeros((interval,2))  #set empty arrays
    p2 = np.zeros((interval,2))
    p_text = np.zeros((interval,2))
    for i in range(0,interval):
        for j in range(0,2):
            if (j%2==0):
                p1[i][j] = x + 0.9 * r * np.cos(separation * i * 3.14 / 180) #point for lines
            else:
                p1[i][j] = y + 0.9 * r * np.sin(separation * i * 3.14 / 180)
    text_offset_x = 10
    text_offset_y = 5
    for i in range(0, interval):
        for j in range(0, 2):
            if (j % 2 == 0):
                p2[i][j] = x + r * np.cos(separation * i * 3.14 / 180)
                p_text[i][j] = x - text_offset_x + 1.2 * r * np.cos((separation) * (i+9) * 3.14 / 180) #point for text labels, i+9 rotates the labels by 90 degrees
            else:
                p2[i][j] = y + r * np.sin(separation * i * 3.14 / 180)
                p_text[i][j] = y + text_offset_y + 1.2* r * np.sin((separation) * (i+9) * 3.14 / 180)  # point for text labels, i+9 rotates the labels by 90 degrees

    #add the lines and labels to the image
    for i in range(0,interval):
        cv2.line(img, (int(p1[i][0]), int(p1[i][1])), (int(p2[i][0]), int(p2[i][1])),(0, 255, 0), 2)
        cv2.putText(img, '{}'.format(int(i*separation)), (int(p_text[i][0]), int(p_text[i][1])), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,0),2,cv2.LINE_AA)

    cv2.imwrite('gauge-{}-calibration.{}'.format(gauge_number, file_type), img)


# commenting this out for now. I will use raw inputs for testing

    # #get user input on min, max, values, and units
    # print('gauge number: {}'.format(gauge_number))
    # min_angle = input('Min angle (lowest possible angle of dial) - in degrees: ') #the lowest possible angle
    # max_angle = input('Max angle (highest possible angle) - in degrees: ') #highest possible angle
    # min_value = input('Min value: ') #usually zero
    # max_value = input('Max value: ') #maximum reading of the gauge
    # units = input('Enter units: ')

    # for testing purposes 
    # GAUGE #3
    min_angle = 45
    max_angle = 315
    min_value = -30
    max_value = 22
    units = "PSI"

    return min_angle, max_angle, min_value, max_value, units, x, y, r

def get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type):

    gray2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Set threshold and maxValue
    thresh = 175
    maxValue = 255

    # old method
    ret, th = cv2.threshold(gray2, thresh, maxValue, cv2.THRESH_BINARY_INV)

    # Adaptive Gaussian thresholding. Let's try:
    th_gaussian = cv2.adaptiveThreshold(gray2,maxValue,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

    # Otsu's thresholding after Gaussian filtering
    blur = cv2.GaussianBlur(gray2,(5,5),0)
    ret_otsu_blur,th_otsu_blur = cv2.threshold(blur,0,maxValue,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    ret_otsu,th_otsu = cv2.threshold(gray2,0,maxValue,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # fig, ax = plt.subplots(2, 2)
    # ax[0, 0].imshow(th) #row=0, col=0
    # ax[1, 0].imshow(th_gaussian) #row=1, col=0
    # ax[0, 1].imshow(th_otsu) #row=0, col=1
    # ax[1, 1].imshow(th_otsu_blur) #row=1, col=1
    # plt.xticks([]),plt.yticks([])
    # plt.show()

    # old thresholding method
    # plt.imshow(th)
    # plt.title('plotting original thresholding w/ v = 175')
    # plt.xticks([]),plt.yticks([])
    # plt.show()

    # plt.imshow(gray2)
    # plt.imshow(th_gaussian)
    # plt.title('plotting gauge w/ gaussian')
    # plt.xticks([]),plt.yticks([])
    # plt.show()

    # # otsu's thresholding 
    # plt.imshow(th_otsu)
    # plt.title('plotting gauge w/ OTSU')
    # plt.xticks([]), plt.yticks([])
    # plt.show()

    # find lines
    height, width = img.shape[:2]
    # print('Height of picture: {}'.format(height))
    # print('Width of picture: {}'.format(width))

    minLineLength = np.around(height*.10)  # min 10% of picture height
    # print('minLineLength is {}'.format(minLineLength))
    maxLineGap = 0

    lines = cv2.HoughLinesP(image=th, rho=3, theta=np.pi / 180, threshold=100,minLineLength = minLineLength, maxLineGap = maxLineGap)  # rho is set to 3 to detect more lines, easier to get more then filter them out later

    # for testing purposes, show all found lines
    # for line in lines:
    #     x1,y1,x2,y2 = line[0,:]
    #     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    #     cv2.imwrite('gauge-{}-lines-test.{}'.format(gauge_number, file_type), img)

    # remove all lines outside a given radius
    final_line_list = []
    #print "radius: %s" %r

    diff1LowerBound = 0.15 #diff1LowerBound and diff1UpperBound determine how close the line should be from the center
    diff1UpperBound = 0.25
    diff2LowerBound = 0.5 #diff2LowerBound and diff2UpperBound determine how close the other point of the line should be to the outside of the gauge
    diff2UpperBound = 1.0

    # Lines are in the format: 
    # [x1, y1, x2, y2]
    for line in lines:
        x1,y1,x2,y2 = line[0,:] # line is [[x1, y1, x2, y2]]

        diff1 = dist_2_pts(x, y, x1, y1)  # x, y is center of circle
        diff2 = dist_2_pts(x, y, x2, y2)  # x, y is center of circle
        #set diff1 to be the smaller (closest to the center) of the two), makes the math easier
        if (diff1 > diff2):
            temp = diff1
            diff1 = diff2
            diff2 = temp
        # check if line is within an acceptable range
        if (((diff1<diff1UpperBound*r) and (diff1>diff1LowerBound*r) and (diff2<diff2UpperBound*r)) and (diff2>diff2LowerBound*r)):
            line_length = dist_2_pts(x1, y1, x2, y2)
            # add to final list
            final_line_list.append([x1, y1, x2, y2])

    #testing only, show all lines after filtering
    for i in range(0,len(final_line_list)):
        x1 = final_line_list[i][0]
        y1 = final_line_list[i][1]
        x2 = final_line_list[i][2]
        y2 = final_line_list[i][3]
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # assumes the first line is the best one
    print(final_line_list)

    x1 = final_line_list[0][0]
    y1 = final_line_list[0][1]
    x2 = final_line_list[0][2]
    y2 = final_line_list[0][3]

    #for testing purposes, show the line overlayed on the original image
    cv2.imwrite('gauge-1-test.jpg', img)
    cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imwrite('gauge-{}-lines-2.{}'.format(gauge_number, file_type), img)

    #find the farthest point from the center to be what is used to determine the angle
    dist_pt_0 = dist_2_pts(x, y, x1, y1)
    dist_pt_1 = dist_2_pts(x, y, x2, y2)
    if (dist_pt_0 > dist_pt_1):
        x_angle = x1 - x
        y_angle = y - y1
    else:
        x_angle = x2 - x
        y_angle = y - y2
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    #np.rad2deg(res) #coverts to degrees

    # print x_angle
    # print y_angle
    # print res
    # print np.rad2deg(res)

    #these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  #in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  #in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  #in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  #in quadrant IV
        final_angle = 270 - res

    #print final_angle

    old_min = float(min_angle)
    old_max = float(max_angle)

    new_min = float(min_value)
    new_max = float(max_value)

    old_value = final_angle

    old_range = (old_max - old_min)
    new_range = (new_max - new_min)
    new_value = (((old_value - old_min) * new_range) / old_range) + new_min

    return new_value

def main():
    gauge_number = 3
    file_type='jpg'

    # image name format 'gauge-#.jpg', for example 'gauge-5.jpg'
    img = cv2.imread('gauge-{}.{}'.format(gauge_number, file_type))
    # calibrate
    min_angle, max_angle, min_value, max_value, units, x, y, r = calibrate_gauge(img, gauge_number, file_type)

    # take calibrationv alue and calculate current gauge value
    val = get_current_value(img, min_angle, max_angle, min_value, max_value, x, y, r, gauge_number, file_type)

    print ("Current reading: {} {}".format(val, units))

if __name__=='__main__':
    main()
    
