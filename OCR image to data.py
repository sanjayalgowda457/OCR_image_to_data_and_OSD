import cv2
import pytesseract as pt
import pandas as pd



pt.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

img=cv2.imread("test file 3.png")
cv2.imshow("img=",img)
cap=cv2.resize(img,None,fx=0.5,fy=0.5)
gray=cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)

adaptive_threshold=cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,85,11)
config ="--psm 3"
text = pt.image_to_data(img, output_type='data.frame')
text = text[text.conf != -1]
print(text.head())

lines = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['text'] \
    .apply(lambda x: ' '.join(list(x))).tolist()
confs = text.groupby(['page_num', 'block_num', 'par_num', 'line_num'])['conf'].mean().tolist()

line_conf = []

for i in range(len(lines)):
    if lines[i].strip():
        line_conf.append((lines[i], round(confs[i], 3)))
       # print
cv2.imshow("img=",img)