#  Brain Tumor Detection using CNN

Detect brain tumors from MRI images using a **Convolutional Neural Network (CNN)** built with **TensorFlow & Keras**.  
This Model demonstrates the full deep learning pipeline: **data preprocessing, augmentation, model training, evaluation, and saving the model**.

---

## Project Structure
brain-tumor-detection-cnn/
├── brain_tumor_cnn.py
├── README.md
├── requirements.txt
├── LICENSE
├── class_indices.json
├── final_brain_tumor_model.h5 (optional, large)
└── brain_tumor_dataset/
├── train/
│ ├── glioma/
│ ├── meningioma/
│ ├── pituitary/
│ └── no_tumor/
└── test/
├── glioma/
├── meningioma/
├── pituitary/
└── no_tumor/


ps://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri).

---

##  Steps Implemented

1. **Importing Libraries** – TensorFlow, Keras, OpenCV, Pandas, NumPy, Matplotlib, Seaborn.  
2. **Loading and Preprocessing Dataset** – read images, label classes, build DataFrames.  
3. **Data Augmentation** – rotation, shifting, zooming, flipping.  
4. **Building CNN Model** – Conv2D, MaxPooling, BatchNormalization, Dense, Dropout layers.  
5. **Training & Validation** – using Adam optimizer and categorical cross-entropy.  
6. **Evaluation & Visualization** – accuracy/loss plots, confusion matrix, classification report.  
7. **Saving Model** – `final_brain_tumor_model.h5` and `class_indices.json`.  

---
## Results

Accuracy: ~95% (depends on dataset and training)
Confusion matrix and classification report generated.
Sample predictions visualized on test images.

##Technologies Used

Python 3.x
TensorFlow / Keras
OpenCV
NumPy & Pandas
Matplotlib & Seaborn
scikit-learn

License:- This project is licensed under the MIT License – see the LICENSE
 file for details.

## Author : Suhani Yadav
B.Tech CSE | Deep Learning Enthusiast | Traditional yet forward-thinking ✨
GitHub: https://github.com/<Suhani-yadav04>


