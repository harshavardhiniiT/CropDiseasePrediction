from flask import Flask, request, render_template
import cv2
import joblib
from skimage.feature import hog

app = Flask(__name__)

model = joblib.load("model_final.pkl")
le = joblib.load("label_encoder.pkl")
scaler = joblib.load("scaler.pkl")

IMG_SIZE = 128

@app.route("/", methods=["GET", "POST"])
def index():
    result = None

    if request.method == "POST":
        file = request.files["image"]
        file.save("test.jpg")

        img = cv2.imread("test.jpg")
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = hog(
            gray,
            orientations=9,
            pixels_per_cell=(8, 8),
            cells_per_block=(2, 2)
        )

        features = scaler.transform([features])
        prediction = model.predict(features)
        result = le.inverse_transform(prediction)[0]

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)