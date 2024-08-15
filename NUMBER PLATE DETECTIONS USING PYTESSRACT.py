# NUMBER PLATE DETECTION USING PYTESSARACT


import cv2
import imutils  # We will need this library to resize our images.
import pytesseract  # We will need this library to extract the license plate text from the detected license plate.

pytesseract.pytesseract.tesseract_cmd =r"C:\Users\VENKATA SURYA\Pycharm\nareshit projects\deep learning"

image = cv2.imread(r"C:\Users\VENKATA SURYA\OneDrive\Documents\jeep.jpg")
resized_image = imutils.resize(image)
cv2.imshow('original image', image)
cv2.waitKey(0)

gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("greyed image", gray_image)
cv2.waitKey(0)

gray_image = cv2.bilateralFilter(gray_image, 11, 17, 17)
cv2.imshow("smoothened image", gray_image)
cv2.waitKey(0)

edged = cv2.Canny(gray_image, 30, 200)
cv2.imshow("edged image", edged)
cv2.waitKey(0)

cnts, new = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
image1 = image.copy()
cv2.drawContours(image1, cnts, -1, (0, 255, 0), 3)
cv2.imshow("contours", image1)
cv2.waitKey(0)

cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:30]
screenCnt = None
image2 = image.copy()
cv2.drawContours(image2, cnts, -1, (0, 255, 0), 3)
cv2.imshow("Top 30 contours", image2)
cv2.waitKey(0)

i = 7
for c in cnts:
    perimeter = cv2.arcLength(c, True)
    approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
    if len(approx) == 4:
        screenCnt = approx
        x, y, w, h = cv2.boundingRect(c)
        new_img = image[y:y + h, x:x + w]
        cv2.imwrite('./' + str(i) + '.png', new_img)
        i += 1
        break

cv2.drawContours(image, [screenCnt], -1, (0, 255, 0), 3)
cv2.imshow("image with detected license plate", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
explanation of code

This code implements a basic number plate detection system using OpenCV and Tesseract OCR. 
The process begins by reading and resizing the input image, followed by converting it to grayscale and applying a bilateral filter to reduce noise. 
Edge detection is then performed using the Canny algorithm to highlight potential contours. 
The contours are detected and sorted, with the largest ones being considered as potential number plates. 
The code approximates the contours to a polygon and checks for a four-sided shape, which is likely a license plate. 
Once detected, the license plate is cropped from the image and saved.
and the final result is displayed with the detected number plate highlighted.
'''