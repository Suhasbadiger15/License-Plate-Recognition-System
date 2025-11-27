# License-Plate-Recognition-System

This project implements an Automatic License Plate Recognition (ALPR) system designed primarily for Indian number plates.  
The system performs image preprocessing, license plate detection, character segmentation, and OCR-based recognition using classical vision + ML techniques.

### Features

- Upload car image as input  
- Grayscale conversion and noise removal  
- Contrast enhancement using CLAHE  
- Vertical Sobel edge detection  
- Region-of-interest detection using variance masking  
- Harris corner detection with region weighting  
- Automatic license plate localization  
- Character segmentation using contour extraction  
- OCR using Tesseract or CNN model  
- Simple GUI built using Gradio  

### Project Pipeline

#### Pre-Processing
- Convert image to grayscale  
- Apply bilateral filter for noise removal  
- Enhance contrast using CLAHE  

#### Plate Detection
- Apply Sobel vertical edge detection  
- Perform binarization using Otsu threshold  
- ROI masking based on variance distribution  
- Filter regions using contour size constraints  
- Use Harris corner detection & dilation  
- Weight regions based on corner density  
- Select best candidate region  
- Extract contour & bounding rectangle for plate  

#### Character Recognition
- Crop and align plate region  
- Character segmentation using morphology  
- Binarization and noise removal  
- OCR using PyTesseract (or CNN classifier)
