import os

# ── FORWARD INSTRUCTIONS TO TENSORFLOW ─────────────────────────────────
# 1. Force TensorFlow to use ONLY the CPU (disables GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Fix potential weight loading issues between TF 2.15 and TF 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ───────────────────────────────────────────────────────────────────────

import uuid
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ── 1. Cancer Types & Class Mappings ──────────────────────────────────
LEVEL1_CLASSES = [
    "Acute Lymphoblastic Leukemia",
    "Brain Cancer",
    "Breast Cancer",
    "Cervical Cancer",
    "Kidney Cancer",
    "Lung and Colon Cancer",
    "Lymphoma",
    "Oral Cancer",
]

LEVEL2_DISPLAY = [
    "ALL - Benign (Healthy Cells)",
    "ALL - Early Stage",
    "ALL - Pre Stage",
    "ALL - Pro (Advanced)",
    "Brain - Glioma",
    "Brain - Meningioma",
    "Brain - Pituitary Tumor",
    "Breast - Benign",
    "Breast - Malignant",
    "Cervix - Dyskeratotic",
    "Cervix - Koilocytotic",
    "Cervix - Metaplastic",
    "Cervix - Parabasal",
    "Cervix - Superficial Intermediate",
    "Kidney - Normal",
    "Kidney - Tumor",
    "Colon - Adenocarcinoma",
    "Colon - Benign Tissue",
    "Lung - Adenocarcinoma",
    "Lung - Benign Tissue",
    "Lung - Squamous Cell Carcinoma",
    "Lymphoma - CLL",
    "Lymphoma - Follicular",
    "Lymphoma - Mantle Cell",
    "Oral - Normal",
    "Oral - Squamous Cell Carcinoma",
]

SUBCLASS_TO_CANCER = {
    0: 0, 1: 0, 2: 0, 3: 0,
    4: 1, 5: 1, 6: 1,
    7: 2, 8: 2,
    9: 3, 10: 3, 11: 3, 12: 3, 13: 3,
    14: 4, 15: 4,
    16: 5, 17: 5, 18: 5, 19: 5, 20: 5,
    21: 6, 22: 6, 23: 6,
    24: 7, 25: 7,
}

# ── 2. FastAPI Setup ───────────────────────────────────────────────────
app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

IMAGE_SIZE = 224


# ── 3. Model Building & Loading ────────────────────────────────────────
def build_model(num_level1=8, num_level2=26):
    base = MobileNetV3Large(
        input_shape=(224, 224, 3),
        include_top=False,
        weights="imagenet",
    )

    x = base.output
    x = layers.GlobalAveragePooling2D()(x)

    # Shared dense block
    shared = layers.Dense(1280)(x)
    shared = layers.BatchNormalization()(shared)
    shared = layers.ReLU()(shared)
    shared = layers.Dropout(0.3)(shared)

    # Level-1 head → Cancer Type
    l1 = layers.Dense(256, activation="relu")(shared)
    l1 = layers.Dropout(0.2)(l1)
    level1_out = layers.Dense(num_level1, activation="softmax", name="level1")(l1)

    # Level-2 head → Subclass
    l2 = layers.Dense(512, activation="relu")(shared)
    l2 = layers.Dropout(0.2)(l2)
    l2 = layers.Dense(256, activation="relu")(l2)
    l2 = layers.Dropout(0.2)(l2)
    level2_out = layers.Dense(num_level2, activation="softmax", name="level2")(l2)

    model = Model(inputs=base.input, outputs=[level1_out, level2_out])
    return model

def load_model(weights_path="models/final_model1.keras"):
    # First, try to load the entire model directly to avoid architecture mismatch
    if os.path.exists(weights_path):
        try:
            loaded_model = tf.keras.models.load_model(weights_path)
            print(f"✅ Loaded full model from {weights_path}")
            return loaded_model
        except Exception as e:
            print(f"⚠️ Could not load full model: {e}")
            print("   Attempting to build model and load weights manually...")
            
    # Fallback to manual build
    fallback_model = build_model()
    if os.path.exists(weights_path):
        try:
            fallback_model.load_weights(weights_path)
            print(f"✅ Loaded weights from {weights_path}")
        except Exception as e:
            print(f"⚠️ Could not load weights: {e}")
            print("   Using randomly initialized model.")
    else:
        print("⚠️ No weights file found. Using randomly initialized model (for testing).")

    return fallback_model

# Load model once at startup
print("Loading model...")
model = load_model()
print("Model ready.\n")


# ── 4. Image Processing & Prediction ───────────────────────────────────
def preprocess_image(image: Image.Image) -> np.ndarray:
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    return img_array

def predict(image: Image.Image):
    img_array = preprocess_image(image)

    # model.predict might return a dict (Keras 3) or list depending on how it was saved
    preds = model.predict(img_array, verbose=0)
    
    # Safely extract the raw arrays
    if isinstance(preds, dict):
        preds_list = list(preds.values())
    else:
        preds_list = preds
        
    # Dynamically assign Level 1 and Level 2 based on the number of classes (8 vs 26)
    if preds_list[0].shape[-1] == len(LEVEL1_CLASSES):
        l1_probs = preds_list[0][0]
        l2_probs = preds_list[1][0]
    else:
        l1_probs = preds_list[1][0]
        l2_probs = preds_list[0][0]

    # Level-1 result
    l1_idx = int(np.argmax(l1_probs))
    l1_label = LEVEL1_CLASSES[l1_idx]
    l1_conf = round(float(l1_probs[l1_idx]) * 100, 2)

    l1_all = {
        cls: round(float(p) * 100, 2)
        for cls, p in zip(LEVEL1_CLASSES, l1_probs)
    }

    # Level-2 result
    l2_idx = int(np.argmax(l2_probs))
    l2_label = LEVEL2_DISPLAY[l2_idx]
    l2_conf = round(float(l2_probs[l2_idx]) * 100, 2)

    # All level-2 probs sorted by confidence (descending)
    l2_all_unsorted = {
        LEVEL2_DISPLAY[i]: round(float(l2_probs[i]) * 100, 2)
        for i in range(len(LEVEL2_DISPLAY))
    }
    l2_all = dict(sorted(l2_all_unsorted.items(), key=lambda x: x[1], reverse=True))

    # Check if level-2 prediction is consistent with level-1
    l2_parent = LEVEL1_CLASSES[int(SUBCLASS_TO_CANCER[l2_idx])]
    consistent = (l2_parent == l1_label)

    # Console Output
    print("\n" + "="*45)
    print("              PREDICTION RESULTS             ")
    print("="*45)
    print(f"🔹 Level-1 (Cancer Type): {l1_label}")
    print(f"   Confidence:            {l1_conf:.2f}%\n")
    print(f"🔹 Level-2 (Subclass)   : {l2_label}")
    print(f"   Confidence:            {l2_conf:.2f}%")
    
    if not consistent:
         print("\n⚠️ Note: The Level-2 subclass does not map to the predicted Level-1 type.")
    print("="*45 + "\n")

    return {
        "level1_label": l1_label,
        "level1_confidence": l1_conf,
        "level1_probs": l1_all,
        "level2_label": l2_label,
        "level2_confidence": l2_conf,
        "level2_probs": l2_all,
        "level2_parent": l2_parent,
        "consistent": consistent,
    }


# ── 5. API Routes ──────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, file: UploadFile = File(...)):
    # Validate file type
    ext = Path(file.filename).suffix.lower()
    allowed = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
    if ext not in allowed:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"request": request, "error": f"Unsupported file type: {ext}"}
        )

    # Save uploaded image
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run prediction
    try:
        image = Image.open(save_path)
        result = predict(image)
    except Exception as e:
        return templates.TemplateResponse(
            request=request, 
            name="index.html", 
            context={"request": request, "error": f"Error processing image: {str(e)}"}
        )

    # Return successful prediction
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "request": request,
            "result": result,
            "image_url": f"/uploads/{filename}",
        }
    )

@app.post("/api/predict")
async def predict_api(file: UploadFile = File(...)):
    ext = Path(file.filename).suffix.lower()
    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    image = Image.open(save_path)
    result = predict(image)
    result["filename"] = filename
    result["image_url"] = f"/uploads/{filename}"
    return result


# ── 6. Run Application ─────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn
    # Optional print to confirm TF configuration
    print("Available devices:", tf.config.list_physical_devices())
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)