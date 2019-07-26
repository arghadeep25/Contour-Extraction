import os
import cv2
import sys
import numpy as np


path = sys.argv[1]
filename = os.path.basename(path)

input_image = cv2.imread(path)
image = cv2.cvtColor(input_image, cv2.COLOR_BGR2GRAY)
image = np.uint8(image)
image = np.expand_dims(image,3)

print('Image Shape:',image.shape)

contours, _ = cv2.findContours(image, 
                                  cv2.RETR_EXTERNAL, 
                                  cv2.CHAIN_APPROX_SIMPLE)

print('Number of Shapes: ',len(contours))

for contour in contours:
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    output = cv2.drawContours(input_image, 
                                    [box], 
                                    0, (0,255,0), 3)

print("Output Shape: ",output.shape)
save_dir = 'results/contour/'+filename
cv2.imwrite(save_dir, output)
cv2.imshow('Output Image',output)
cv2.waitKey(0)
cv2.destroyAllWindows()