import os

# ── FORWARD INSTRUCTIONS TO TENSORFLOW ─────────────────────────────────
# 1. Force TensorFlow to use ONLY the CPU (disables GPU)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# 2. Fix potential weight loading issues between TF 2.15 and TF 2.16+
os.environ["TF_USE_LEGACY_KERAS"] = "1"
# ───────────────────────────────────────────────────────────────────────

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV3Large

# ── Level-1: 8 Cancer Types ───────────────────────────────────────────
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

# ── Level-2: 26 Subclasses (folder names) ─────────────────────────────
LEVEL2_CLASSES = [
    "all_benign", "all_early", "all_pre", "all_pro",
    "brain_glioma", "brain_menin", "brain_tumor",
    "breast_benign", "breast_malignant",
    "cervix_dyk", "cervix_koc", "cervix_mep", "cervix_pab", "cervix_sfi",
    "kidney_normal", "kidney_tumor",
    "colon_aca", "colon_bnt", "lung_aca", "lung_bnt", "lung_scc",
    "lymph_cll", "lymph_fl", "lymph_mcl",
    "oral_normal", "oral_scc",
]

# ── Human-readable names for Level-2 ──────────────────────────────────
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

# ── Subclass → Cancer Type mapping ────────────────────────────────────
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

# ── Dataset folder structure ──────────────────────────────────────────
DATASET_STRUCTURE = {
    "ALL": {
        "level1": 0,
        "subs": {"all_benign": 0, "all_early": 1, "all_pre": 2, "all_pro": 3},
    },
    "Brain Cancer": {
        "level1": 1,
        "subs": {"brain_glioma": 4, "brain_menin": 5, "brain_tumor": 6},
    },
    "Breast Cancer": {
        "level1": 2,
        "subs": {"breast_benign": 7, "breast_malignant": 8},
    },
    "Cervical Cancer": {
        "level1": 3,
        "subs": {
            "cervix_dyk": 9, "cervix_koc": 10, "cervix_mep": 11,
            "cervix_pab": 12, "cervix_sfi": 13,
        },
    },
    "Kidney Cancer": {
        "level1": 4,
        "subs": {"kidney_normal": 14, "kidney_tumor": 15},
    },
    "Lung and Colon Cancer": {
        "level1": 5,
        "subs": {
            "colon_aca": 16, "colon_bnt": 17,
            "lung_aca": 18, "lung_bnt": 19, "lung_scc": 20,
        },
    },
    "Lymphoma": {
        "level1": 6,
        "subs": {"lymph_cll": 21, "lymph_fl": 22, "lymph_mcl": 23},
    },
    "Oral Cancer": {
        "level1": 7,
        "subs": {"oral_normal": 24, "oral_scc": 25},
    },
}

# ── Build Model ────────────────────────────────────────────────────────
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

# ── Load Model ─────────────────────────────────────────────────────────
def load_model(weights_path="models/final_model1.keras"):
    # First, try to load the entire model directly to avoid architecture mismatch
    if os.path.exists(weights_path):
        try:
            model = tf.keras.models.load_model(weights_path)
            print(f"✅ Loaded full model from {weights_path}")
            return model
        except Exception as e:
            print(f"⚠️ Could not load full model: {e}")
            print("   Attempting to build model and load weights manually...")
            
    # Fallback to manual build
    model = build_model()
    if os.path.exists(weights_path):
        try:
            model.load_weights(weights_path)
            print(f"✅ Loaded weights from {weights_path}")
        except Exception as e:
            print(f"⚠️ Could not load weights: {e}")
            print("   Using randomly initialized model.")
    else:
        print("⚠️ No weights file found. Using randomly initialized model (for testing).")

    return model

# ── Predict and Print ──────────────────────────────────────────────────
def predict_and_print(model, image_tensor):
    """
    Takes a preprocessed image tensor, runs it through the model, 
    and prints formatted prediction details to the console.
    """
    # Ensure the image has a batch dimension (1, 224, 224, 3)
    if len(image_tensor.shape) == 3:
        image_tensor = np.expand_dims(image_tensor, axis=0)

    # Get predictions
    preds_level1, preds_level2 = model.predict(image_tensor, verbose=0)

    # Extract Level-1 details
    l1_idx = np.argmax(preds_level1[0])
    l1_conf = preds_level1[0][l1_idx] * 100
    l1_name = LEVEL1_CLASSES[l1_idx]

    # Extract Level-2 details
    l2_idx = np.argmax(preds_level2[0])
    l2_conf = preds_level2[0][l2_idx] * 100
    l2_name = LEVEL2_DISPLAY[l2_idx]

    # Print results to console
    print("\n" + "="*45)
    print("              PREDICTION RESULTS             ")
    print("="*45)
    print(f"🔹 Level-1 (Cancer Type): {l1_name}")
    print(f"   Confidence:            {l1_conf:.2f}%\n")
    print(f"🔹 Level-2 (Subclass)   : {l2_name}")
    print(f"   Confidence:            {l2_conf:.2f}%")
    
    # Optional check: Verify if Level-1 and Level-2 predictions logically match
    expected_l1_idx = SUBCLASS_TO_CANCER.get(l2_idx)
    if expected_l1_idx != l1_idx:
         print("\n⚠️ Note: The Level-2 subclass does not map to the predicted Level-1 type.")
    print("="*45 + "\n")

    return l1_idx, l2_idx

# ── Example Usage ──────────────────────────────────────────────────────
if __name__ == "__main__":
    # Check what devices TF sees (should only list CPU now)
    print("Available devices:", tf.config.list_physical_devices())
    
    # Load the model
    my_model = load_model()
    
    # Create a dummy image (Replace this with your actual image loading/preprocessing)
    # Shape should match your model input: (224, 224, 3)
    dummy_image = np.random.rand(224, 224, 3)
    
    # Run prediction and print
    predict_and_print(my_model, dummy_image)