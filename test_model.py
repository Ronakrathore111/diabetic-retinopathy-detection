import keras
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd

# =========================
# PATH CONFIGURATION
# =========================
MODEL_PATH = "models/final1.h5"

DATASET_TRAIN_PATH = "data/DR/train"   # Used only to extract class names
IMG_PATH = r"data/DR/test/1_Mild DR/Mild_DR_65.png"   # Single test image
TEST_FOLDER = r"data/DR/test/3_Proliferate DR"        # Folder test
EXPORT_CSV = True                                     # Save CSV or not

# =========================
# LOAD MODEL (CRITICAL FIX)
# =========================
print(f"\n Loading model from: {MODEL_PATH}")

model = keras.models.load_model(
    MODEL_PATH,
    compile=False,
    safe_mode=False
)

print(" Model loaded successfully ✅")

# =========================
# LOAD CLASS NAMES
# =========================
class_names = sorted([
    d for d in os.listdir(DATASET_TRAIN_PATH)
    if os.path.isdir(os.path.join(DATASET_TRAIN_PATH, d))
])

print(f" Class labels detected: {class_names}")

# =========================
# SINGLE IMAGE PREDICTION
# =========================
def predict_single_image(img_path):
    img = image.load_img(img_path, target_size=(380, 380))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = x / 255.0

    pred = model.predict(x)
    class_idx = np.argmax(pred)
    confidence = np.max(pred) * 100
    predicted_class = class_names[class_idx]

    print("\n Prediction Results:")
    for i, c in enumerate(class_names):
        print(f"{c:25s}: {pred[0][i]*100:.2f}%")

    print(f"\n🔍 Final Prediction: {predicted_class} ({confidence:.2f}% confidence)")

    # Display image + probability bar chart
    plt.figure(figsize=(10, 5))

    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}\n({confidence:.1f}% Confidence)")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    bars = plt.barh(class_names, pred[0] * 100)
    bars[class_idx].set_color("orange")
    plt.xlabel("Confidence (%)")
    plt.title("Class Probability Distribution")
    plt.xlim(0, 100)

    plt.tight_layout()
    plt.show()

# =========================
# FOLDER PREDICTION
# =========================
def predict_folder(folder_path):
    results = []
    print(f"\n Predicting all images in folder: {folder_path}")

    for file in os.listdir(folder_path):
        if file.lower().endswith((".png", ".jpg", ".jpeg")):
            img_path = os.path.join(folder_path, file)

            img = image.load_img(img_path, target_size=(380, 380))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = x / 255.0

            pred = model.predict(x, verbose=0)
            class_idx = np.argmax(pred)
            confidence = np.max(pred) * 100
            predicted_class = class_names[class_idx]

            results.append({
                "Filename": file,
                "Predicted_Class": predicted_class,
                "Confidence (%)": round(confidence, 2)
            })

    df = pd.DataFrame(results)

    print("\n Prediction Summary:")
    print(df.head())

    if EXPORT_CSV:
        csv_path = os.path.join(folder_path, "predictions.csv")
        df.to_csv(csv_path, index=False)
        print(f"\n Predictions saved to: {csv_path}")

    return df

# =========================
# MAIN MENU
# =========================
mode = input("\nChoose mode — (1) Single image  or  (2) Folder batch: ").strip()

if mode == "1":
    predict_single_image(IMG_PATH)
elif mode == "2":
    predict_folder(TEST_FOLDER)
else:
    print("❌ Invalid selection. Please choose 1 or 2.")
