# License-Plate-Recognition-System

This project implements an Automatic License Plate Recognition (ALPR) system designed primarily for Indian number plates.
The system uses image preprocessing, plate detection, character segmentation, and OCR-based recognition.

***Features***

 Upload car image as input

Grayscale conversion and noise removal

 Contrast enhancement using CLAHE

 Vertical Sobel edge detection

 Region-of-interest detection based on variance masking

 Harris corner detection + weighting

 Automatic license plate localization

 Character segmentation using contour extraction

OCR using Tesseract or CNN model

 Simple GUI using Gradio

 ***Project Pipeline***
 
**Pre-Processing**

Convert image to grayscale

Bilateral filter for noise removal

Contrast enhancement using CLAHE

**Plate Detection**

Sobel vertical edge detection

Binarization using Otsu threshold

ROI masking based on variance distribution

Region filtering based on contour size constraints

Harris corner detection & dilation

Weighting regions based on corner density

Select best candidate region

Extract contour & bounding rectangle for plate

**Character Recognition**

Plate cropping & alignment (phase adjustment)

Character segmentation using morphology

Binarization & noise removal

OCR using PyTesseract
