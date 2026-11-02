```python
import gradio as gr
import numpy as np
import tensorflow as tf
import torch

from sklearn.linear_model import LogisticRegression

from transformers import (
    BlipProcessor,
    BlipForConditionalGeneration
)

from PIL import Image


# ==============================
# MACHINE LEARNING
# Student Pass / Fail Prediction
# ==============================

x = np.array([
    [1], [2], [3], [4], [5],
    [6], [7], [8], [9], [10]
])

y = np.array([
    0, 0, 0, 0, 0,
    1, 1, 1, 1, 1
])

ml = LogisticRegression()
ml.fit(x, y)


def student_prediction(hours):
    if hours is None:
        return "Please enter study hours"

    prediction = ml.predict([[hours]])[0]

    if prediction == 1:
        return "PASS"
    else:
        return "FAIL"


# ==============================
# DEEP LEARNING
# Handwritten Digit Recognition
# ==============================

(x_train, y_train), (x_test, y_test) = (
    tf.keras.datasets.mnist.load_data()
)

x_train = x_train[:20000] / 255
y_train = y_train[:20000]

dl_model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(28, 28)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax"),
])

dl_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

dl_model.fit(
    x_train,
    y_train,
    epochs=3,
    verbose=1
)


def digit_prediction(image):
    if image is None:
        return "Please upload an image"

    image = image.convert("L")

    # Convert to numpy
    image = np.array(image)

    # Invert white background
    if image.mean() > 127:
        image = 255 - image

    # Find the digit
    coords = np.argwhere(image > 50)

    if coords.size == 0:
        return "No digit detected"

    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)

    # Crop the digit
    image = image[y_min:y_max + 1, x_min:x_max + 1]

    # Resize cropped digit to 28x28
    image = Image.fromarray(image).resize((28, 28))

    image = np.array(image) / 255.0
    image = image.reshape(1, 28, 28)

    prediction = dl_model.predict(
        image,
        verbose=0
    )

    digit = np.argmax(prediction)

    return f"Recognized Digit: {digit}"


# ==============================
# GENERATIVE AI
# AI Image Captioning
# ==============================

processor = BlipProcessor.from_pretrained(
    "SalesForce/blip-image-captioning-base"
)

gem_model = BlipForConditionalGeneration.from_pretrained(
    "SalesForce/blip-image-captioning-base"
)

device = "cuda" if torch.cuda.is_available() else "cpu"

gem_model.to(device)


def generate_caption(image):
    if image is None:
        return "Please upload an image!"

    image = image.convert("RGB")

    inputs = processor(
        images=image,
        return_tensors="pt"
    ).to(device)

    with torch.no_grad():
        output = gem_model.generate(
            **inputs,
            max_new_tokens=50
        )

    caption = processor.decode(
        output[0],
        skip_special_tokens=True
    )

    return caption


# ==============================
# GRADIO APPLICATION
# ==============================

with gr.Blocks(title="AI Demonstration") as app:

    gr.Markdown("""
    # AI & MACHINE LEARNING USING PYTHON
    ### Combined Demonstration of ML, Deep Learning and Generative AI

    Explore three different Artificial Intelligence applications.
    """)

    # Machine Learning Tab
    with gr.Tab("Machine Learning"):

        gr.Markdown(
            "## Student Pass / Fail Prediction"
        )

        hours = gr.Number(
            label="Enter Study Hours",
            value=5
        )

        ml_button = gr.Button(
            "Predict Result"
        )

        ml_output = gr.Textbox(
            label="ML Prediction"
        )

        ml_button.click(
            fn=student_prediction,
            inputs=hours,
            outputs=ml_output
        )

    # Deep Learning Tab
    with gr.Tab("Deep Learning"):

        gr.Markdown(
            "## Handwritten Digit Recognition"
        )

        digit_image = gr.Image(
            type="pil",
            label="Upload a Handwritten Digit"
        )

        dl_button = gr.Button(
            "Recognize Digit"
        )

        dl_output = gr.Textbox(
            label="DL Prediction"
        )

        dl_button.click(
            fn=digit_prediction,
            inputs=digit_image,
            outputs=dl_output
        )

    # Generative AI Tab
    with gr.Tab("Generative AI"):

        gr.Markdown(
            "## AI Image Captioning"
        )

        gem_image = gr.Image(
            type="pil",
            label="Upload an Image"
        )

        gem_button = gr.Button(
            "Generate Image Caption"
        )

        gem_output = gr.Textbox(
            label="AI Generated Caption",
            lines=3
        )

        gem_button.click(
            fn=generate_caption,
            inputs=gem_image,
            outputs=gem_output
        )


app.launch(share=True)
```

