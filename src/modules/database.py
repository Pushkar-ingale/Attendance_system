import os
import numpy as np
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from PIL import Image
from src.modules.db_helper import initialize_database, insert_or_update_student

# Load the pre-trained FaceNet model
FACENET_MODEL_PATH = "D:\PBL_Project_2024\Attendance_system\model\facenet_keras.h5"
facenet_model = load_model(FACENET_MODEL_PATH)
print("FaceNet model loaded successfully.")

# Initialize the MTCNN face detector
detector = MTCNN()

# Ensure the database is initialized
initialize_database()

# Helper Function: Preprocess a face for FaceNet
def preprocess_face(face_image, required_size=(160, 160)):
    """
    Resizes and normalizes a face image for the FaceNet model.
    :param face_image: Face image as a NumPy array.
    :param required_size: Target size for FaceNet input.
    :return: Preprocessed face image ready for embedding.
    """
    image = Image.fromarray(face_image).resize(required_size)
    face_array = np.asarray(image).astype("float32") / 255.0
    face_array = np.expand_dims(face_array, axis=0)  # Add batch dimension
    return face_array

# Helper Function: Detect and extract a face from an image
def extract_face(image):
    """
    Detects and extracts the largest face from the image.
    :param image: Input image as a NumPy array.
    :return: Cropped face image as a NumPy array, or None if no face is detected.
    """
    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None

    # Choose the face with the largest bounding box (most prominent face)
    face = max(faces, key=lambda x: x['box'][2] * x['box'][3])  # Width * Height
    x, y, width, height = face['box']
    x, y = abs(x), abs(y)  # Ensure no negative values
    return image[y:y + height, x:x + width]

# Main Function: Capture faces and store embeddings
def capture_faces(prnno, name, images):
    """
    Captures face embeddings from a set of images and stores the information in the database.
    :param prnno: Unique student identifier (PRN number).
    :param name: Name of the individual.
    :param images: List of 30 images as NumPy arrays.
    """
    embeddings = []

    for image in images:
        # Step 1: Detect and extract face from the image
        face = extract_face(image)
        if face is None:
            print("No face detected in one of the images. Skipping...")
            continue

        # Step 2: Preprocess the face for FaceNet
        face_array = preprocess_face(face)

        # Step 3: Generate the face embedding
        embedding = facenet_model.predict(face_array)
        embeddings.append(embedding[0])  # Extract embedding vector

    if len(embeddings) == 0:
        print("No valid embeddings were generated. Exiting...")
        return

    # Step 4: Compute the super embedding (mean of all embeddings)
    super_embedding = np.mean(embeddings, axis=0)

    # Step 5: Store the information in the database
    embedding_blob = super_embedding.tobytes()
    insert_or_update_student(prnno, name, embedding_blob)

