import cv2
import pytesseract
from PIL import Image

image = cv2.imread('imgs/label1.png')
#image = cv2.imread('imgs/label3.jpg')
orig = image.copy()

image_grey = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
cv2.imwrite('output/grey.png', image_grey)

image_sobel = cv2.Sobel(image_grey, cv2.CV_8U, 1, 1, ksize=3)
cv2.imwrite('output/sobel.png', image_sobel)

image_threshold = cv2.threshold(image_sobel, 0, 255, cv2.THRESH_OTSU+cv2.THRESH_BINARY)
cv2.imwrite('output/threshold.png', image_threshold[1])

element = cv2.getStructuringElement(cv2.MORPH_RECT, (17,3))
image_threshold = cv2.morphologyEx(image_threshold[1], cv2.MORPH_CLOSE, element)
contours = cv2.findContours(image_threshold, 0, 1)
rects = []
for c in contours[0]:
    if len(c) < 100:
        continue
    poly = cv2.approxPolyDP(c, 3, True)
    x,y,w,h = cv2.boundingRect(poly)
    if h>w:
        continue
    rects.append(((x,y),(x+w,y+h)))

for r in rects:
    cv2.rectangle(image, r[0], r[1], (0,255,0), 1)
cv2.imwrite('output/rects.png', image)

texts = []
for i,crop_area in enumerate(rects):
    cropped_image = orig[crop_area[0][1]:crop_area[1][1],crop_area[0][0]:crop_area[1][0]]

    cropped_pil_image = Image.fromarray(cropped_image)
    for _ in range(5):
        text = pytesseract.image_to_string(cropped_pil_image)
        if len(text) > 0:
            break
        s = cropped_pil_image.size
        cropped_pil_image = cropped_pil_image.resize((s[0]*2,s[1]*2))
    cropped_pil_image.save('output/cropped-%d.png'%i, 'png')
    print('-'*20)
    print(text)
    texts.append(text)

# Search for calories
