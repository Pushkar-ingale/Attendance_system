import os
import sqlite3
import numpy as np
from PIL import Image
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1
import base64
from io import BytesIO
from src.modules.db_helper import insert_or_update_student

# Path to the SQLite database
DATABASE_PATH = r"database/attendance.db"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True, device='cpu')  # Use GPU for faster face detection
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().cpu()

# Step 1: Detect and extract the largest face from the image
def extract_face(image):
    """
    Detects and extracts the largest face from the image.
    :param image: Input image as a PIL Image.
    :return: Cropped face image as a NumPy array or None if no face is detected.
    """
    boxes, _ = mtcnn.detect(image)
    if boxes is None or len(boxes) == 0:
        return None

    # Extract the largest face
    face = max(boxes, key=lambda box: (box[2] - box[0]) * (box[3] - box[1]))
    x1, y1, x2, y2 = map(int, face)
    return image.crop((x1, y1, x2, y2))

# Step 2: Preprocess face image for FaceNet
def preprocess_face(face_image, required_size=(160, 160)):
    """
    Resizes and normalizes a face image for FaceNet.
    :param face_image: Face image as a PIL Image.
    :param required_size: Target size for FaceNet input.
    :return: Preprocessed face tensor ready for embedding generation.
    """
    face_image = face_image.resize(required_size).convert('RGB')
    face_tensor = torch.from_numpy(np.array(face_image).astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
    return face_tensor.cpu()

# Step 3: Capture face embeddings and store them in the database
def capture_faces(prnno, name, images):
    """
    Captures face embeddings for a student and stores them in the database.
    :param prnno: Unique student identifier (PRN number).
    :param name: Name of the student.
    :param images: List of images in base64 format.
    """
    embeddings = []

    for image in images:
        # Decode and preprocess each image
        image_data = Image.open(BytesIO(base64.b64decode(image.split(",")[1])))
        face = extract_face(image_data)
        if face is None:
            print("No face detected in one of the images. Skipping...")
            continue

        face_tensor = preprocess_face(face)
        with torch.no_grad():
            embeddings.append(facenet_model(face_tensor).cpu().numpy().squeeze())

    if len(embeddings) == 0:
        print("No valid embeddings generated. Exiting...")
        return

    # Compute the mean embedding for the student
    mean_embedding = np.mean(embeddings, axis=0).tobytes()
    insert_or_update_student(prnno, name, mean_embedding)
    print(f"Student {name} (PRN: {prnno}) embeddings captured and stored successfully.")
