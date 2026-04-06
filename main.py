import os
import uuid
import shutil
import numpy as np
from pathlib import Path
from PIL import Image
from typing import List

# Disable oneDNN custom operations for consistency
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large, ResNet50
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ==========================================
# MODEL PATH CONFIGURATION
# ==========================================
MOBILENET_PATH = "models/final_model1.keras" 
RESNET_PATH    = "models/resnet50_model.keras"
# ==========================================

LEVEL1_CLASSES = [
    "Acute Lymphoblastic Leukemia", "Brain Cancer", "Breast Cancer", "Cervical Cancer",
    "Kidney Cancer", "Lung and Colon Cancer", "Lymphoma", "Oral Cancer",
]

LEVEL2_DISPLAY = [
    "ALL - Benign (Healthy Cells)", "ALL - Early Stage", "ALL - Pre Stage", "ALL - Pro (Advanced)",
    "Brain - Glioma", "Brain - Meningioma", "Brain - Pituitary Tumor",
    "Breast - Benign", "Breast - Malignant",
    "Cervix - Dyskeratotic", "Cervix - Koilocytotic", "Cervix - Metaplastic", "Cervix - Parabasal", "Cervix - Superficial Intermediate",
    "Kidney - Normal", "Kidney - Tumor",
    "Colon - Adenocarcinoma", "Colon - Benign Tissue",
    "Lung - Adenocarcinoma", "Lung - Benign Tissue", "Lung - Squamous Cell Carcinoma",
    "Lymphoma - CLL", "Lymphoma - Follicular", "Lymphoma - Mantle Cell",
    "Oral - Normal", "Oral - Squamous Cell Carcinoma",
]

SUBCLASS_TO_CANCER = {
    0: 0, 1: 0, 2: 0, 3: 0, 4: 1, 5: 1, 6: 1, 7: 2, 8: 2, 9: 3, 10: 3, 11: 3, 12: 3, 13: 3,
    14: 4, 15: 4, 16: 5, 17: 5, 18: 5, 19: 5, 20: 5, 21: 6, 22: 6, 23: 6, 24: 7, 25: 7,
}

app = FastAPI()
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

IMAGE_SIZE = 224

def build_hierarchical_model(base_type="mobilenet"):
    if base_type == "mobilenet":
        base = MobileNetV3Large(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
    else:
        base = ResNet50(input_shape=(224, 224, 3), include_top=False, weights="imagenet")

    x = layers.GlobalAveragePooling2D()(base.output)
    shared = layers.Dense(1280, activation="relu")(x)
    
    l1 = layers.Dense(256, activation="relu")(shared)
    level1_out = layers.Dense(len(LEVEL1_CLASSES), activation="softmax", name="cancer_type")(l1)

    l2 = layers.Dense(512, activation="relu")(shared)
    l2 = layers.Dense(256, activation="relu")(l2)
    level2_out = layers.Dense(len(LEVEL2_DISPLAY), activation="softmax", name="subclass")(l2)

    return Model(inputs=base.input, outputs=[level1_out, level2_out])

def get_model(path, model_type):
    if not os.path.exists(path):
        print(f"Model file not found: {path}. Building untrained architecture.")
        return build_hierarchical_model(model_type)

    if path.endswith(".weights.h5"):
        print(f"Detected weights file for {model_type}. Building architecture and loading weights...")
        try:
            m = build_hierarchical_model(model_type)
            m.load_weights(path)
            return m
        except Exception as e:
            print(f"Failed to load .weights.h5 for {model_type}: {e}")
            return build_hierarchical_model(model_type)

    try:
        print(f"Attempting to load full model from {path}...")
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"Could not load full model from {path}: {e}")
        try:
            m = build_hierarchical_model(model_type)
            m.load_weights(path)
            return m
        except Exception as e2:
            print(f"Could not load weights: {e2}")
            return build_hierarchical_model(model_type)

print("--- Loading Models ---")
loaded_models = {
    "mobilenet": get_model(MOBILENET_PATH, "mobilenet"),
    "resnet": get_model(RESNET_PATH, "resnet")
}
print("--- Models Ready ---")

def predict(image: Image.Image, model_choice: str):
    target_model = loaded_models.get(model_choice, loaded_models["mobilenet"])
    
    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img_array = np.array(img, dtype=np.float32)
    img_array = np.expand_dims(img_array, axis=0)
    
    if model_choice == "mobilenet":
        img_array = img_array / 255.0
    elif model_choice == "resnet":
        img_array = img_array / 255.0
    else:
        img_array /= 255.0

    print(f"\n[TensorFlow] Triggering forward pass using {model_choice.upper()} model...")
    preds = target_model.predict(img_array, verbose=1)
    
    # STRICT extraction (no guessing)
    if isinstance(preds, dict):
        l1_probs = preds["cancer_type"][0]
        l2_probs = preds["subclass"][0]
    else:
        # fallback (in case model loads differently)
        l1_probs, l2_probs = preds
        l1_probs = l1_probs[0]
        l2_probs = l2_probs[0]

    # sanity check (VERY IMPORTANT)
    print("L1 sum:", np.sum(l1_probs))
    print("L2 sum:", np.sum(l2_probs))
    print("L1 max:", np.max(l1_probs))
    print("L2 max:", np.max(l2_probs))

    l1_idx = int(np.argmax(l1_probs))
    l2_idx = int(np.argmax(l2_probs))
    
    # Raw unrounded probabilities directly from the model
    l1_all = {cls: float(p) * 100 for cls, p in zip(LEVEL1_CLASSES, l1_probs)}
    l2_all_unsorted = {LEVEL2_DISPLAY[i]: float(l2_probs[i]) * 100 for i in range(len(LEVEL2_DISPLAY))}
    l2_all = dict(sorted(l2_all_unsorted.items(), key=lambda x: x[1], reverse=True))

    l2_parent = LEVEL1_CLASSES[int(SUBCLASS_TO_CANCER[l2_idx])]
    
    level1_label = LEVEL1_CLASSES[l1_idx]
    level1_conf = float(l1_probs[l1_idx]) * 100
    
    level2_label = LEVEL2_DISPLAY[l2_idx]
    level2_conf = float(l2_probs[l2_idx]) * 100
    level1_conf_rounded = round(level1_conf, 2)
    level2_conf_rounded = round(level2_conf, 2)
    # Print to console directly from the prediction function
    print("="*40)
    print(f"PREDICTION RESULTS ({model_choice.upper()})")
    print(f"Level 1 (Cancer Type): {level1_label} [{level1_conf:.6f}%]")
    print(f"Level 2 (Subclass)   : {level2_label} [{level2_conf:.6f}%]")
    print("="*40 + "\n")
    
    return {
        "level1_label": level1_label,
        "level1_confidence": level1_conf_rounded,
        "level1_probs": l1_all,
        "level2_label": level2_label,
        "level2_confidence": level2_conf_rounded,
        "level2_probs": l2_all,
        "level2_parent": l2_parent,
        "consistent": (l2_parent == level1_label),
        "model_used": model_choice
    }

@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={"request": request}
    )

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, file: UploadFile = File(...), model_choice: str = Form("mobilenet")):
    
    # SAFETY FIX: Ensure form data precisely matches the dictionary keys 
    model_choice = model_choice.lower().strip()
    if model_choice not in ["mobilenet", "resnet"]:
        model_choice = "mobilenet"

    ext = Path(file.filename).suffix.lower()
    if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
        return templates.TemplateResponse(
            request=request, name="index.html", context={"request": request, "error": f"Invalid type: {ext}"}
        )

    filename = f"{uuid.uuid4().hex}{ext}"
    save_path = UPLOAD_DIR / filename
    with open(save_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        image = Image.open(save_path)
        # result logs are now printed directly inside this predict() function
        result = predict(image, model_choice)
        
        return templates.TemplateResponse(
            request=request, name="index.html", context={"request": request, "result": result, "image_url": f"/uploads/{filename}"}
        )
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "error": str(e)})


@app.post("/predict_batch", response_class=HTMLResponse)
async def predict_batch_route(request: Request, files: List[UploadFile] = File(...), model_choice: str = Form("mobilenet")):
    
    # SAFETY FIX
    model_choice = model_choice.lower().strip()
    if model_choice not in ["mobilenet", "resnet"]:
        model_choice = "mobilenet"

    if not files:
        return templates.TemplateResponse(
            request=request, name="index.html", context={"request": request, "error": "No files uploaded."}
        )
    
    batch_results = []
    
    for file in files:
        ext = Path(file.filename).suffix.lower()
        if ext not in {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}:
            continue
            
        filename = f"{uuid.uuid4().hex}{ext}"
        save_path = UPLOAD_DIR / filename
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
            
        try:
            image = Image.open(save_path)
            # result logs are now printed directly inside this predict() function
            result = predict(image, model_choice)
            
            batch_results.append({
                "original_name": file.filename,
                "image_url": f"/uploads/{filename}",
                "result": result
            })
            
        except Exception as e:
            print(f"Error processing {file.filename}: {e}")

    return templates.TemplateResponse(
        request=request, 
        name="index.html", 
        context={
            "request": request, 
            "batch_results": batch_results, 
            "message": f"Successfully processed {len(batch_results)} images."
        }
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)