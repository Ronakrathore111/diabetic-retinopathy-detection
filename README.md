# 👁️ Diabetic Retinopathy Detection (Deep Learning + Grad-CAM)

A **local deep learning application** that detects **Diabetic Retinopathy (DR)** stages from retinal fundus images using deep learning models.  
The project supports **model-based prediction**, **Grad-CAM visualization**, and **PDF report generation** for academic and learning purposes.

---

# 📌 Overview

Diabetic Retinopathy (DR) is a leading cause of blindness among diabetic patients.  
Early detection is essential, and deep learning models can help analyze retinal fundus images efficiently.

This project demonstrates how CNN-based models can be used for DR stage classification in an **offline, local environment**.

This project provides:

✔ Automatic DR Stage Classification  
✔ Grad-CAM heatmaps for model explainability  
✔ Multi-image batch processing  
✔ PDF report generation  
✔ Local interactive interface  
✔ Modular and extensible deep learning pipeline  

---

# ⭐ Features

- ✔ Upload and analyze retinal images locally  
- ✔ Predict DR severity: **Healthy → Severe DR**  
- ✔ Grad-CAM heatmaps for visual interpretability  
- ✔ Batch prediction for multiple images  
- ✔ Generate PDF reports with prediction details  
- ✔ Clean and user-friendly interface for experimentation  

---

# 🧠 Model Architecture

This project uses transfer learning with:

- EfficientNet  
- ResNet  
- Custom CNN models  

Training pipeline includes:

- Image normalization  
- Data augmentation  
- Class imbalance handling  
- Softmax-based multi-class classification  

### Models Used
- best_model.h5  
- final_model.h5  
- final1.h5  

> ⚠️ Models are used for **educational and demonstration purposes only**.

---

# 📂 Project Structure


diabetic-retinopathy/
│
├── dashboard.py
├── gradcam.py
├── train.py
├── evaluate.py
├── evaluate_model.py
├── test_model.py
├── utils.py
├── requirements.txt
├── README.md
└── models/
## 🔧 Installation & Setup (Local Only)

# 1️⃣ Clone the repository
git clone https://github.com/ronarathore111/diabetic-retinopathy-detection.git
cd diabetic-retinopathy-detection

# 2️⃣ Create a virtual environment
python -m venv venv

# 3️⃣ Activate the virtual environment (Windows)
venv\Scripts\activate

# 4️⃣ Install required dependencies
pip install -r requirements.txt

# 5️⃣ Run the application locally
python dashboard.py

---

## 🔥 Why Grad-CAM?

Grad-CAM (Gradient-weighted Class Activation Mapping) helps in understanding how deep learning models make decisions by visualizing important regions in retinal images.

It helps to analyze:

- Which regions of the retina influence predictions  
- Whether the model focuses on relevant pathological features  
- How confident the model is in its classification  
- The interpretability of deep learning predictions  

This improves transparency and trust in AI-based medical image analysis.

---

## 🖼 Grad-CAM Visualization

For each retinal image, the system performs the following steps:

1. Predicts the Diabetic Retinopathy (DR) class  
2. Generates a Grad-CAM heatmap  
3. Displays the original image and the heatmap together  
4. Includes the visualization in the generated report  

This allows better understanding of model behavior and decision-making.

---

## 📄 PDF Report Generation

The application supports automatic generation of PDF reports that include:

- Predicted DR stage  
- Confidence scores  
- Original retinal image  
- Grad-CAM heatmap visualization  
- Optional patient information fields  

This feature is intended **only for academic and learning purposes**.

## 📦 Requirements

tensorflow-cpu==2.13.0
numpy==1.24.3
pandas
opencv-python-headless
matplotlib
scikit-learn
tqdm
streamlit
pillow
scipy
gdown
fpdf

DR Classification Labels
0 — Healthy  
1 — Mild DR  
2 — Moderate DR  
3 — Proliferative DR  
4 — Severe DR  

👤 Author

Ronak Rathore

GitHub → https://github.com/ronarathore111
