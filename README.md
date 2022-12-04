# Document Extraction

This project is focused on information extraction from Offical Indian Documents like Aadhar, PAN, Cheque etc. This was done using techniques like **Object detection (yolov7)** and **OCR (Tesseract and Transformer based)**.

# Requirements
- [requirements.txt](https://github.com/mallapraveen/Document-Extraction/blob/main/requirements.txt)

# Table Of Contents
-  [In a Nutshell](#in-a-nutshell)
-  [In Details](#in-details)
-  [Future Work](#future-work)
-  [Contributing](#contributing)
-  [Acknowledgments](#acknowledgments)

# In a Nutshell

1. We gathered the dataset(images) from Internet. For Aadhar, we generated our own dataset using the aadahr templates available online and using opencv techniques we manipulated and came up with fake data and generated the dataset for aadhar. For PAN and Cheques, we collected the images from the Internet.
2. We annotated the documents with labels(Name, DOB, Address, Aadhar No., PAN no.) using the platform [Roboflow](https://roboflow.com/)
3. We then trained the model using [yolov7](https://github.com/WongKinYiu/yolov7)
4. We used the cropped images from object detection and sent those to OCR for convertion to text. [Tesseract](https://github.com/tesseract-ocr/tesseract) & [Transformer Based](https://huggingface.co/microsoft/trocr-small-printed)

# In Details
```
├──  Notebooks
│    └── *.ipny - here are the notebooks used for dev & testing.
│
│
├──  Sample Images  
│    └── *  - here are the sample and generated images of the documents.
│ 
│
├──  src
│    └── api  - hereis the flask endpoints api's for the models
│    └── generated_aadhar - here are the genrated aadhar images using scripts.
│    └── input - here's the inputs required for the scripts
│    └── output - here's the outputs generated by the script
│    └── preprocessing - here's scripts used for generating aadhar images
│
├──  generated_aadhar
│    └── *  - here are the genrated aadhar images using scripts.

```

# Future Work

This can be extended to other documents like Death Certificate, driving license etc.

# Contributing

Any kind of enhancement or contribution is welcomed.