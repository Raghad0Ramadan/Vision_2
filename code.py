#Done by Raghad Ramadan
#student num :32109303032

import cv2
import numpy as np


image = cv2.imread("img_1.png")
img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

templite = cv2.imread("temp.png", cv2.IMREAD_GRAYSCALE)

w, h = templite.shape[::-1]

tem_d = cv2.matchTemplate(img_gray,templite,cv2.TM_CCOEFF_NORMED)
threshold = 0.8
loc = np.where(tem_d >= threshold)
for pt in zip(*loc[::-1]):
 cv2.rectangle(image, pt, (pt[0] + w, pt[1] + h), (0,255,0), 2)


cv2.imshow("tem_dec",image)
cv2.imwrite("Matching_1.jpg", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

