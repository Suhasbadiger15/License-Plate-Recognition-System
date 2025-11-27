import cv2
import numpy as np
import pytesseract

# pytesseract configuration
pytesseract.pytesseract.tesseract_cmd = r"/usr/bin/tesseract"  # change if needed
OCR_CONFIG = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Preprocessing function
def preprocess_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.bilateralFilter(gray, 11, 17, 17)

    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(blur)

    return enhanced

# Detect possible license plate region
def detect_plate_region(image):
    pre = preprocess_image(image)

    sobel = cv2.Sobel(pre, cv2.CV_64F, 1, 0, ksize=3)
    sobel = cv2.convertScaleAbs(sobel)

    _, thresh = cv2.threshold(sobel, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 3))
    morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    possible = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect = w / float(h)

        if 2.0 < aspect < 6.5 and w > 100 and h > 20:
            possible.append((x, y, w, h))

    if len(possible) == 0:
        return None

    best = max(possible, key=lambda r: r[2] * r[3])
    x, y, w, h = best
    plate = image[y:y+h, x:x+w]
    return plate

# Character segmentation
def segment_characters(plate_img):
    gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    kernel = np.ones((3, 3), np.uint8)
    morph = cv2.morphologyEx(th, cv2.MORPH_DILATE, kernel)

    contours, _ = cv2.findContours(morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    char_imgs = []
    for c in sorted(contours, key=lambda c: cv2.boundingRect(c)[0]):
        x, y, w, h = cv2.boundingRect(c)
        if h > 25 and h < 120 and w > 10:
            char_imgs.append(morph[y:y+h, x:x+w])

    return char_imgs

# OCR recognition
def ocr_plate(plate_img):
    chars = segment_characters(plate_img)
    result = ""
    for ch in chars:
        resized = cv2.resize(ch, (32, 32))
        text = pytesseract.image_to_string(resized, config=OCR_CONFIG)
        result += text.strip()
    return result

# Full pipeline
def detect_and_recognize_license_plate(image):
    plate_img = detect_plate_region(image)
    if plate_img is None:
        return "Plate not detected", image
    text = ocr_plate(plate_img)
    return text, plate_img
