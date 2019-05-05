import numpy as np
import cv2

right_image = cv2.imread('right.jpg',0)          
left_image = cv2.imread('left.jpg',0)

det_akaze = cv2.AKAZE_create()

# using SIFT detect the keypoints and descriptors
key_p1, descriptor1 = det_akaze.detectAndCompute(right_image, None)
key_p2, descriptor2 = det_akaze.detectAndCompute(left_image, None)

# use BFMatcher to match descriptor
bf_match = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches_found = bf_match.match(descriptor1,descriptor2)

# Based on their distance sort the matches
matches_found = sorted(matches_found, key = lambda x:x.distance)
# print(matches_found)

source_points = np.float32([ key_p1[matches_found[m].queryIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
destination_points = np.float32([ key_p2[matches_found[m].trainIdx].pt for m in range(0, 20) ]).reshape(-1,1,2)
# print(source_points)
# print(destination_points)

homograph_val, mask = cv2.findHomography(source_points, destination_points, cv2.RANSAC)

# Use homography
height, width = right_image.shape
im1Reg_right = cv2.warpPerspective(right_image, homograph_val, (width, height))
cv2.imwrite("Output_wraped.jpg", im1Reg_right)
left_resized = cv2.resize(left_image, (right_image.shape[1], right_image.shape[0]))

# 'bitwise OR' operation
# result = cv2.bitwise_or(left_resized, im1Reg_right)

result = cv2.addWeighted(left_resized, 0.8, im1Reg_right, 0.8, 5)
# To have no anomilies in the merged picture use cv2.addweighted

cv2.imwrite("Output_merged.jpg", result)

cv2.waitKey(0)
cv2.destroyAllWindows()
