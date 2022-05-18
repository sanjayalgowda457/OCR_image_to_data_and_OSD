import cv2
import matplotlib.pyplot as plt
import numpy as np
import easyocr
reader = easyocr.Reader(["en"])

# Threshold image and determine connected components
img_bgr = cv2.imread("C:/Users/sanjay a l/Desktop/data science/OCR tesseract image todata/test-file2.png")
img_gray = cv2.cvtColor(img_bgr[35:115, 30:], cv2.COLOR_BGR2GRAY)
ret, img_bin = cv2.threshold(img_gray, 195, 255, cv2.THRESH_BINARY_INV)
retval, labels = cv2.connectedComponents(255 - img_bin, np.zeros_like(img_bin), 8)
fig, axs = plt.subplots(4)
axs[0].imshow(img_gray, cmap="gray")
axs[0].set_title("grayscale")
axs[1].imshow(img_bin, cmap="gray")
axs[1].set_title("thresholded")
axs[2].imshow(labels, vmin=0, vmax=retval - 1, cmap="tab20b")
axs[2].set_title("connected components")

# Find and process individual characters
OCR_out = ""
all_img_chars = np.zeros((labels.shape[0], 0), dtype=np.uint8)
labels_xmin = [np.argwhere(labels == i)[:, 1].min() for i in range(0, retval)]
# Process the labels (connected components) from left to right
for i in np.argsort(labels_xmin):
    label_yx = np.argwhere(labels == i)
    label_ymin = label_yx[:, 0].min()
    label_ymax = label_yx[:, 0].max()
    label_xmin = label_yx[:, 1].min()
    label_xmax = label_yx[:, 1].max()
    # Characters are large blobs that don't border the top/bottom edge
    if label_yx.shape[0] > 250 and label_ymin > 0 and label_ymax < labels.shape[0]:
        img_char = img_bin[:, label_xmin - 3 : label_xmax + 3]
        all_img_chars = np.hstack((all_img_chars, img_char))
        # Use EasyOCR on single char (pytesseract performs poorly on single characters)
        OCR_out += reader.recognize(img_char, detail=0)[0]
axs[3].imshow(all_img_chars, cmap="gray")
axs[3].set_title("individual characters")
fig.show()

print("Thruth:  6UAE005X0721295")
print("OCR out: " + OCR_out)