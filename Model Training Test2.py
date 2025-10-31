import json
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as kimage
from pathlib import Path

MODEL_PATH = "image_classifier.h5"
CLASS_INDICES_PATH = "class_indices.json"   # save this after training (see below)

def load_class_names():
    # Fallback: stable sort of subfolders if JSON isn't available
    if Path(CLASS_INDICES_PATH).exists():
        with open(CLASS_INDICES_PATH, "r", encoding="utf-8") as f:
            class_indices = json.load(f)                # {"cat":0,"dog":1}
        # invert to index->name list in correct order
        idx2name = [None]*len(class_indices)
        for name, idx in class_indices.items():
            idx2name[idx] = name
        return idx2name, ("binary" if len(idx2name)==2 else "categorical")
    else:
        # Last resort (not recommended)
        classes = sorted([d.name for d in Path("dataset/train").iterdir() if d.is_dir()])
        return classes, ("binary" if len(classes)==2 else "categorical")

def predict_image(image_path):
    image_path = Path(image_path)
    if not image_path.exists():
        print(f"Error: File not found at path: {image_path}")
        return

    # Load model
    model = tf.keras.models.load_model(MODEL_PATH)
    class_names, class_mode = load_class_names()

    # Determine input size dynamically: (None, H, W, C)
    _, H, W, _ = model.input_shape

    # Use Keras loader to ensure RGB and consistent resizing
    img = kimage.load_img(str(image_path), target_size=(H, W))  # default is RGB
    arr = kimage.img_to_array(img)

    # IMPORTANT: match training preprocessing
    # If you used tf.keras.applications.* preprocess_input, call it here instead of /255.0
    arr = arr / 255.0

    # Predict
    x = np.expand_dims(arr, axis=0)
    preds = model.predict(x, verbose=0)

    if class_mode == "binary":
        prob = float(preds[0][0])
        predicted_idx = int(prob >= 0.5)           # tune threshold later
        predicted_class = class_names[predicted_idx]
        print(f"{predicted_class} (p={prob:.3f})")
        title = f"{predicted_class} (p={prob:.2f})"
    else:
        predicted_idx = int(np.argmax(preds[0]))
        predicted_class = class_names[predicted_idx]
        conf = float(np.max(preds[0]))
        print(f"{predicted_class} (p={conf:.3f})")
        title = f"{predicted_class} (p={conf:.2f})"

    # Optional: display
    import matplotlib.pyplot as plt
    plt.imshow(img)          # already RGB
    plt.title(f"The model has determined: {title}")
    plt.axis("off")
    plt.show()

predict_image("img/test/fishtest2.jpeg")