from PIL import Image
import pytesseract
import cv2

pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Get information about orientation and script detection
#print(pytesseract.image_to_osd(Image.open('test.png')))
text = pytesseract.image_to_osd(r'test.png')
