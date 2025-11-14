import os
import warnings
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory
from PIL import Image
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from utils import CompatibleDepthwiseConv2D, strip_groups_from_h5


# -------------------------------
# Flask App
# -------------------------------
app = Flask(__name__)
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------------------
# Load Vitamin Classification Model
# -------------------------------
cleaned_path = strip_groups_from_h5("vitamins_classification_model.h5")

model = load_model(
    cleaned_path,
    compile=False,
    custom_objects={'DepthwiseConv2D': CompatibleDepthwiseConv2D}
)

vitamin_classes = ['VitaminA', 'VitaminB', 'VitaminC', 'VitaminD', 'VitaminE']


# -------------------------------
# Routes
# -------------------------------
@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        if "image" not in request.files:
            return jsonify({"status": "error", "message": "No file uploaded"})

        file = request.files["image"]
        filename = file.filename
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)

        # Load + preprocess
        img = Image.open(filepath).convert("RGB")
        img = img.resize((224, 224))

        x = image.img_to_array(img)
        x = np.expand_dims(x, axis=0)

        pred = np.argmax(model.predict(x), axis=1)
        result = vitamin_classes[pred[0]]

        return jsonify({
            "status": "success",
            "prediction": result,
            "filename": filename
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route("/uploads/<filename>")
def serve_uploaded(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/result")
def result_page():
    filename = request.args.get("filename")
    prediction = request.args.get("prediction")
    return render_template("result.html", filename=filename, prediction=prediction)


# -------------------------------
# Run App
# -------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
