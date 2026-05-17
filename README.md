# WORKSHOP---5-License-Plate-Detection-using-OpenCV-and-Haar-Cascade-Classifier


# PYTHON CODE 


```

import cv2
import matplotlib.pyplot as plt
import os
import urllib.request

# -------------------------------------------------------------
# Step 1: Read and display the input image
# -------------------------------------------------------------
image_path = 'img.jpg'

image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(
        "Image not found. Please check the image_path variable."
    )

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title("Original Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 2: Convert to grayscale
# -------------------------------------------------------------
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

plt.imshow(gray, cmap='gray')
plt.title("Grayscale Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 3: Preprocessing
# -------------------------------------------------------------
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

equalized = cv2.equalizeHist(blurred)

plt.imshow(equalized, cmap='gray')
plt.title("Preprocessed Image")
plt.axis('off')
plt.show()

# -------------------------------------------------------------
# Step 4: Load Haar Cascade
# -------------------------------------------------------------
cascade_path = 'haarcascade_frontalface_default.xml'

# Download cascade file automatically
if not os.path.exists(cascade_path):

    print("Cascade file not found. Downloading...")

    url = "https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml"

    urllib.request.urlretrieve(url, cascade_path)

    print("Cascade file downloaded successfully!")

# Load classifier
face_cascade = cv2.CascadeClassifier(cascade_path)

# -------------------------------------------------------------
# Step 5: Detect Faces
# -------------------------------------------------------------
faces = face_cascade.detectMultiScale(
    equalized,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30)
)

print(f"Total Faces Detected: {len(faces)}")

# -------------------------------------------------------------
# Step 6: Draw Rectangles and Save Faces
# -------------------------------------------------------------
output = image.copy()

save_dir = "Detected_Faces"

os.makedirs(save_dir, exist_ok=True)

for i, (x, y, w, h) in enumerate(faces):

    cv2.rectangle(
        output,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        3
    )

    face_crop = image[y:y+h, x:x+w]

    save_path = f"{save_dir}/face_{i+1}.jpg"

    cv2.imwrite(save_path, face_crop)

if len(faces) > 0:
    print(f"{len(faces)} face(s) saved in '{save_dir}' folder.")
else:
    print("No faces detected.")

# -------------------------------------------------------------
# Step 7: Display Final Output
# -------------------------------------------------------------
plt.imshow(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
plt.title("Detected Faces")
plt.axis('off')
plt.show()

```


# OUTPUT 
### DETECTED FACES
<img width="452" height="486" alt="image" src="https://github.com/user-attachments/assets/ca821967-6795-4c1a-86b9-647688793f8c" />

