""" List of important parameters to tweak and try in case you dont get output
1. Frame rate (frameRate)
2. Number of matches required
3. Feaature distance
4. Number of frames for mini panorama per batch
5. Number of frames to skip after stitching a small panorama
6.Size of input images if program starts to lag hould be reduced

"""
# import modules and libraries

import cv2
import numpy as np
import glob
import imutils
from modules import *

# from modules import draw_matches, cropping, crop, warpImages

# the helper functions are defined in modules.py
# 1.draw_matches() 2.warpImages() 3. crop() 4. cropping() 5. trim()

# Main program begins here

# Define input and output paths
# video source, contains path of input video

"""This part of program should be uncommented if we want to use webcam or camera to record video"""

"""# Define the codec and create VideoWriter object
name = 'output.mp4'
cap = cv2.VideoCapture(0)
fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
out = cv2.VideoWriter(name, fourcc, 5.0, (int(cap.get(3)), int(cap.get(4))))
# loop runs if capturing has been initialized.
while (True):
    # reads frames from a camera
    # ret checks return at each frame
    ret, frame = cap.read()
    if ret == True:
    # output the frame
        out.write(frame)

    # The window showing the operated video stream
        cv2.imshow('frame', frame)


    # Wait for 'a' key to stop the program
        if cv2.waitKey(1) & 0xFF == ord('a'):
            break
    else:
        break

# Close the window / Release webcam
cap.release()

# After we release our webcam, we also release the output
out.release()
"""

input_path = "/Users/akshayacharya/Desktop/Panorama/Bazinga/Test images for final/Highfps2fps/out_test5.mp4"

# Convert video to frames and store in another directory
path_to_frames = "/Users/akshayacharya/Desktop/Panorama/Bazinga/Test images for final/Frames/"

count = 1
vidcap = cv2.VideoCapture("out.mp4")
sec = 0

print('hi')
############    ADJUST FRAME RATE HERE     ##########
frameRate = 0.33


############    IMPORTANT PARAMETER        ##########

def getFrame(sec):
    vidcap.set(cv2.CAP_PROP_POS_MSEC, sec * 1000)
    hasFrames, image = vidcap.read()
    if hasFrames:
        path = path_to_frames + str(count) + ".jpg"
        print(path)
        cv2.imwrite(path, image)  # save frame as JPG file
    return hasFrames


success = getFrame(sec)
while success:
    count = count + 1
    sec = sec + frameRate
    sec = round(sec, 2)
    success = getFrame(sec)

vidcap.release()
# End of video to frame conversion

# Read the frames from input path
frame_input_path = "/Users/akshayacharya/Desktop/Panorama/Bazinga/Test images for final/Frames/*.jpg"

# Define whatever variables necessary

input_img = glob.glob(frame_input_path)
img_path = sorted(input_img)
# Resize the input images if its too big and pano is starting to lag
for i in range(0, len(img_path)):
    img = cv2.imread(img_path[i])
    img = cv2.resize(img, (1920, 1080))
    cv2.imwrite(img_path[i], img)

tmp = img_path[0]
flag = True
pano = []
i = 1
count = 0
indices = []
k = 1

# First set of panoramas

while i < len(img_path):
    indices.append(i)
    print(i)
    count += 1
    if flag:
        img1 = cv2.imread(tmp, cv2.COLOR_BGR2GRAY)
        img2 = cv2.imread(img_path[i], cv2.COLOR_BGR2GRAY)
        flag = False
    # img1 = cv2.resize(img1, (0, 0), fx=1, fy=1)
    img2 = cv2.imread(img_path[i], cv2.COLOR_BGR2GRAY)
    # img2 = cv2.resize(img2, (0, 0), fx=1, fy=1)

    # Adjust number of features to look for between images. Default is 2000, change it if needed adn see what happens
    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(img1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2, None)

    # Create a BFMatcher object.
    # It will find all of the matching keypoints on two images
    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    # Find matching points
    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    all_matches = []
    for m, n in matches:
        all_matches.append(m)

    # Finding the best matches
    good = []
    for m, n in matches:
        # vary this distance and see what happens
        ##########     PARAMETER       #######
        if m.distance < 0.9 * n.distance:
            #####################################
            good.append(m)

    ##########     PARAMETER       #######
    MIN_MATCH_COUNT = 15
    #####################################

    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result = warpImages(img2, img1, M)

    i += 1
    ###### PARAMETER ####### This decides how many frames to stitch at a time
    if count % 5 == 0:
        ########################
        i += 5  ### This parameter decides how many frames to skip
        count = 0
        # result = trim(img1)         #These are three cropping mechanisms
        result = crop(img1)
        # result = cropping(img1)
        # result = result[:, 25:]

        # Output path for where the smaller panoramas are to be written
        cv2.imwrite(f"Test images for final/Smaller panoramas/frame{k}.jpg", result)

        k += 1   # index of the smaller panorama
        indices = []
        try:
            img1 = cv2.imread(img_path[i])
            # i = i + 1
            ###### RESIZE THE NEXT INPUT IMAGE IF NEEDED ####
            img1 = cv2.resize(img1, (1080, 1920))
            cv2.imshow("Stitch", result)
            cv2.waitKey(0)
        except:
            continue

# This ends the smaller panoramas in batches as specified

#Now if exactly no images are left and the batch and incremetn leads exactly to the last frame
if len(indices) == 0:
    indices = [0]
    j = 100
# Not sure why i added this
if len(indices) == 7:
    print('Hi')
    indices = [0]
    j = 100 # This means theres nothing lef to do so directly it will eventually go to just stacking

#IF indices length is not 0, ie, a few images are left and need to be stitched

if indices[0] != 0:
    print('Going to stitch last panorama')
    i = 0
    print(indices)
    j = indices[i]
    temp = img_path[j]


#If only one image is left
if j == (len(img_path) - 1):
    img_1 = cv2.imread(temp)  #This is the only image left and so last panorama is just this

#Stitch the last panorama
i = 1
flag1 = True
while i < len(indices):
    if flag1:
        img_1 = cv2.imread(temp, cv2.COLOR_BGR2GRAY)
        j = indices[i]
        img_2 = cv2.imread(img_path[j], cv2.COLOR_BGR2GRAY)
        flag1 = False
    img_1 = cv2.resize(img1, (0, 0), fx=1, fy=1)
    img_2 = cv2.imread(img_path[i], cv2.COLOR_BGR2GRAY)
    img_2 = cv2.resize(img2, (0, 0), fx=1, fy=1)

    orb = cv2.ORB_create(nfeatures=2000)

    keypoints1, descriptors1 = orb.detectAndCompute(img_1, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img_2, None)

    bf = cv2.BFMatcher_create(cv2.NORM_HAMMING)

    matches = bf.knnMatch(descriptors1, descriptors2, k=2)

    all_matches = []
    for m, n in matches:
        all_matches.append(m)

    # Finding the best matches
    good = []
    for m, n in matches:
        if m.distance < 0.9 * n.distance:    #PARAMETER
            good.append(m)

    MIN_MATCH_COUNT = 20   #PARAMETER

    if len(good) > MIN_MATCH_COUNT:
        # Convert keypoints to an argument for findHomography
        src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

        # Establish a homography
        M, _ = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        result1 = warpImages(img_2, img_1, M)
        img_1 = result1

    i += 1

#Small panorama is stitched for the last one

if j != 100:
    img_1 = cv2.resize(img_1, (1080, 1920))
    cv2.imwrite(f"Test images for final/Smaller panoramas/frame{k}.jpg", img_1)
    #This panorama will just be last image or the slast small panorama based on how many frames were left
    cv2.imshow("Last pano", img_1)
    cv2.waitKey(0)
#All panoramas are written and are ready to be stacked


#Comment from here if you dont want it to be stacked and be left as individual small panoramas

input_path = "/Users/akshayacharya/Desktop/Panorama/Bazinga/Test images for final/Smaller panoramas/*.jpg"
output_path = "/Users/akshayacharya/Desktop/Panorama/Bazinga/Output/panout.jpg"

list_images = glob.glob(input_path)
list_sorted = sorted(list_images)

images = []
#Resize all images to same size before stacking
for image in list_sorted:
    img = cv2.imread(image)
    img = cv2.resize(img, (1400, 720))
    images.append(img)

final_image = cv2.hconcat(images)
print ( np.shape(final_image))
final_image = cv2.resize(final_image,(0,0), fx = 1.0, fy = 1.0)
cv2.imshow("Final stacked panorama", final_image)
cv2.waitKey(0)
cv2.imwrite(output_path, final_image)
