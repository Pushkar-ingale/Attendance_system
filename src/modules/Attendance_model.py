import os
import sqlite3
import csv
import numpy as np
from datetime import datetime
from tensorflow.keras.models import load_model
from mtcnn import MTCNN
from PIL import Image

# Load the FaceNet model and MTCNN detector
FACENET_MODEL_PATH = "models/facenet_keras.h5"
DATABASE_PATH = "database/attendance.db"

facenet_model = load_model(FACENET_MODEL_PATH)
detector = MTCNN()

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

# Helper Function: Detect and extract all faces from an image
def extract_faces(image):
    """
    Detects and extracts all faces from the image.
    :param image: Input image as a NumPy array.
    :return: List of cropped face images.
    """
    faces = detector.detect_faces(image)
    extracted_faces = []

    for face in faces:
        x, y, width, height = face['box']
        x, y = abs(x), abs(y)  # Ensure no negative values
        face_crop = image[y:y + height, x:x + width]
        extracted_faces.append(face_crop)

    return extracted_faces

# Helper Function: Match embedding with database
def match_embedding_with_db(embedding):
    """
    Matches the given face embedding with entries in the SQLite database.
    :param embedding: Face embedding as a NumPy array.
    :return: Matched student information or None if no match is found.
    """
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()

    # Retrieve all records from the database
    cursor.execute("SELECT prnno, name, embedding FROM students")
    records = cursor.fetchall()

    for prnno, name, db_embedding_blob in records:
        db_embedding = np.frombuffer(db_embedding_blob, dtype=np.float32)

        # Compute cosine similarity
        cosine_similarity = np.dot(embedding, db_embedding) / (
            np.linalg.norm(embedding) * np.linalg.norm(db_embedding)
        )
        if cosine_similarity > 0.8:  # Threshold for matching
            conn.close()
            return {"prnno": prnno, "name": name}

    conn.close()
    return None

# Main Function: Mark Attendance
def mark_attendance(filepath, subject, class_no, department, year):
    """
    Marks attendance by analyzing the given image and matching detected faces.
    :param filepath: Path to the image file.
    :param subject: Subject for attendance.
    :param class_no: Class number.
    :param department: Department of the students.
    :param year: Academic year.
    """
    # Check if the file exists
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return

    # Read the image
    image = np.asarray(Image.open(filepath).convert("RGB"))

    # Extract all faces from the image
    faces = extract_faces(image)
    if len(faces) == 0:
        print("No faces detected in the image.")
        return

    # Initialize the attendance list
    attendance_list = []

    for face in faces:
        # Preprocess the face and get the embedding
        preprocessed_face = preprocess_face(face)
        embedding = facenet_model.predict(preprocessed_face)[0]

        # Match the embedding with the database
        student = match_embedding_with_db(embedding)
        if student:
            attendance_list.append(student)
            print(f"Matched: {student['name']} (PRN: {student['prnno']})")
        else:
            print("No match found for a detected face.")

    # Record attendance in a CSV file
    if attendance_list:
        # Define the CSV file path
        attendance_dir = "attendance_records"
        os.makedirs(attendance_dir, exist_ok=True)
        csv_filename = f"{attendance_dir}/attendance_{subject}_{class_no}_{department}_{year}.csv"

        # Write to the CSV file
        file_exists = os.path.exists(csv_filename)
        with open(csv_filename, mode="a", newline="") as csvfile:
            writer = csv.writer(csvfile)

            # Write the header if the file is being created
            if not file_exists:
                writer.writerow(["PRN No", "Name", "Subject", "Class", "Department", "Year", "Timestamp"])

            # Write each attendance record
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            for student in attendance_list:
                writer.writerow([student["prnno"], student["name"], subject, class_no, department, year, timestamp])

        print(f"Attendance marked and saved to {csv_filename}")
    else:
        print("No matches found. Attendance not marked.")

# Example Usage
# if __name__ == "__main__":
#     # Replace with your test values
#     mark_attendance(
#         filepath="test_images/classroom.jpg",
#         subject="Mathematics",
#         class_no="10",
#         department="Computer Science",
#         year="2024"
#     )
