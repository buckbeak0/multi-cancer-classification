import os
import uuid
import shutil
import numpy as np
from pathlib import Path
from PIL import Image

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input as preprocess_mobilenet
from tensorflow.keras.applications.resnet50 import preprocess_input as preprocess_resnet

from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ================= CONFIG =================
MOBILENET_PATH = "models/final_model1.keras"
RESNET_PATH    = "models/resnet50_model.keras"
IMAGE_SIZE = 224

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
    0:0,1:0,2:0,3:0,4:1,5:1,6:1,7:2,8:2,9:3,10:3,11:3,12:3,13:3,
    14:4,15:4,16:5,17:5,18:5,19:5,20:5,21:6,22:6,23:6,24:7,25:7
}

# ================= APP =================
app = FastAPI()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

templates = Jinja2Templates(directory="templates")
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# ================= LOAD MODELS =================
def load_model(path):
    if not os.path.exists(path):
        raise Exception(f"Model not found: {path}")
    print("Loading:", path)
    return tf.keras.models.load_model(path, compile=False)

loaded_models = {
    "mobilenet": load_model(MOBILENET_PATH),
    "resnet": load_model(RESNET_PATH)
}

# ================= PREDICT =================
def predict(image, model_choice):
    model = loaded_models[model_choice]

    img = image.convert("RGB").resize((IMAGE_SIZE, IMAGE_SIZE))
    img = np.array(img, dtype=np.float32)
    img = np.expand_dims(img, axis=0)

    # --- MODEL-SPECIFIC PREPROCESSING ---
    if model_choice == "mobilenet":
        # TRADITIONAL RESCALING: Scale pixels to [0, 1]
        img = img / 255.0
    else:
        # RESNET: Keep the working BGR/Mean-Subtraction logic
        img = preprocess_resnet(img)

    preds = model.predict(img, verbose=0)

    if isinstance(preds, dict):
        l1_probs = preds["cancer_type"][0]
        l2_probs = preds["subclass"][0]
    else:
        l1_probs, l2_probs = preds
        l1_probs = l1_probs[0]
        l2_probs = l2_probs[0]

    l1_idx = int(np.argmax(l1_probs))
    l2_idx = int(np.argmax(l2_probs))
    
    # ================= CONSOLE DIAGNOSTICS =================
    print("\n" + "="*55)
    print(f"🔍 TF PREDICTION | MODEL: {model_choice.upper()}")
    print("-" * 55)
    print(f"➔ Predicted L1       : {LEVEL1_CLASSES[l1_idx]}")
    print(f"➔ L1 Raw Confidence  : {l1_probs[l1_idx]:.8f}")
    print(f"➔ Predicted Subtype  : {LEVEL2_DISPLAY[l2_idx]}")
    print(f"➔ Pixel Range Used   : {np.min(img):.4f} to {np.max(img):.4f}")
    print("="*55 + "\n")

    l1_all = [(cls, float(p)*100) for cls, p in zip(LEVEL1_CLASSES, l1_probs)]
    l2_all = [(LEVEL2_DISPLAY[i], float(l2_probs[i])*100) for i in range(len(LEVEL2_DISPLAY))]

    l1_all.sort(key=lambda x: x[1], reverse=True)
    l2_all.sort(key=lambda x: x[1], reverse=True)

    return {
        "level1_label": LEVEL1_CLASSES[l1_idx],
        "level1_confidence": round(float(l1_probs[l1_idx])*100, 2),
        "level1_probs": l1_all,
        "level2_label": LEVEL2_DISPLAY[l2_idx],
        "level2_confidence": round(float(l2_probs[l2_idx])*100, 2),
        "level2_probs": l2_all,
        "level2_parent": LEVEL1_CLASSES[SUBCLASS_TO_CANCER[l2_idx]],
        "consistent": (LEVEL1_CLASSES[l1_idx] == LEVEL1_CLASSES[SUBCLASS_TO_CANCER[l2_idx]]),
        "model_used": model_choice
    }

# ================= ROUTES =================
@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse(request=request, name="index.html", context={"request": request})

@app.post("/predict", response_class=HTMLResponse)
async def predict_route(request: Request, file: UploadFile = File(...), model_choice: str = Form(...)):
    try:
        model_choice = model_choice.lower().strip()
        filename = f"{uuid.uuid4().hex}.jpg"
        path = UPLOAD_DIR / filename
        with open(path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        image = Image.open(path)
        result = predict(image, model_choice)
        return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "result": result, "image_url": f"/uploads/{filename}"})
    except Exception as e:
        return templates.TemplateResponse(request=request, name="index.html", context={"request": request, "error": str(e)})

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)