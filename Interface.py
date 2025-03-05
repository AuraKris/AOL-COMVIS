import tkinter as tk
from tkinter import filedialog
from tkinter import Label
from PIL import Image, ImageTk
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.densenet import preprocess_input

# Load the model
model = load_model('DenseNet201_model_segmented_fully_89.keras')

# Class names based on your categories
class_names = ['normal', 'adenocarcinoma', 'large.cell.carcinoma', 'squamous.cell.carcinoma']

def segment_lung(image_path):
    # Step 1: Read image in grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Step 3: Apply Gaussian Blur and Otsu's Thresholding
    blur = cv2.GaussianBlur(img, (5, 5), 0)
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Step 4: Find contours and keep only the largest components (lungs)
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(binary_mask)
    cv2.drawContours(mask, contours, -1, 255, thickness=cv2.FILLED)
    
    # Step 5: Remove noise with morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask_cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)  # Remove small artifacts
    
    # Step 6: Create a white background
    white_background = np.full_like(img, 255)  # White background (same size as the original)

    # Step 7: Extract lung regions and replace the rest with white
    lung_segment = cv2.bitwise_and(img, img, mask=mask_cleaned)  # Keep lungs
    result = white_background.copy()  # Start with a white image
    result[mask_cleaned > 0] = lung_segment[mask_cleaned > 0]  # Replace lung regions
    
    # Step 8: Resize for consistency (optional)
    result_resized = cv2.resize(result, (224, 224))
    
    return result_resized

def refine_segment_lung(image_path):
    # Step 1: Read image in grayscale
    # img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = segment_lung(image_path)

    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # Adjust clipLimit and tile size as needed
    enhanced_img = clahe.apply(img)
    
    # Step 2: Apply Otsu's thresholding to create a binary mask
    blur = cv2.GaussianBlur(enhanced_img,(9,9),0)
    _, binary_mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Step 3: Extract outer contours of the lung
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    draw = cv2.drawContours(binary_mask, contours, -1, 255, 3)
    
    # # Step 5: Apply the mask to the original grayscale image
    lung_segment = cv2.bitwise_not(binary_mask)
    
    white_background = np.full_like(enhanced_img, 255)  # Fill image with white pixels
    
    lung_segment2 = cv2.bitwise_and(enhanced_img, enhanced_img, mask=lung_segment)
    
    background_white = cv2.bitwise_not(white_background, white_background, mask=lung_segment2)

    result = background_white.copy()  # Start with a white image
    result[lung_segment > 0] = lung_segment2[lung_segment > 0]  # Replace lung regions
    
    # Resize for consistency
    # result = cv2.resize(result, (224, 224))
    return result

# Function to make a prediction
def classify_image():
    global img_path
    processed_img = refine_segment_lung(img_path)
    # processed_img = np.repeat(processed_img, 3, axis=-1) 
    processed_img = cv2.cvtColor(processed_img, cv2.COLOR_GRAY2RGB)
    processed_img = np.expand_dims(processed_img, axis=0)
    processed_img = preprocess_input(processed_img)
    if processed_img is None:
        result_label.config(text="Error: Could not process image.")
        return

    # Make predictions with the three separate inputs
    prediction = model.predict(processed_img)
    
    # shape_label.config(image=Image.fromarray(processed_img))
    # shape_label.config(image=Image.fromarray(processed_img[0]))
    
    predicted_class = np.argmax(prediction, axis=1)[0]
    confidence = prediction[0][predicted_class]  # Confidence score for the predicted class
    
    # Update result label with prediction and confidence
    result_label.config(
        text=f"Prediction: {class_names[predicted_class]} (Confidence: {confidence:.2%})"
    )

# Function to load an image
def load_image():
    global img_path
    img_path = filedialog.askopenfilename()
    if img_path:
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img = ImageTk.PhotoImage(img)
        panel.config(image=img)
        panel.image = img
        shape_label.config(text="")
        result_label.config(text="")

# Setting up Tkinter window
window = tk.Tk()
window.title("Lung Cancer Classifier")
window.geometry("800x600")

# Load Image Button
btn_load = tk.Button(window, text="Load Image", command=load_image)
btn_load.pack()

# Display Panel for Image
panel = Label(window)
panel.pack()

# Classify Button
btn_classify = tk.Button(window, text="Classify", command=classify_image)
btn_classify.pack()

shape_label = Label(window, text="", font=("Arial", 16))
shape_label.pack()

# Result Label
result_label = Label(window, text="", font=("Arial", 16))
result_label.pack()

# Run the GUI loop
window.mainloop()
