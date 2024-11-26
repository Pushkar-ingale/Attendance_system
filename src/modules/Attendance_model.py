# import os
# import sqlite3
# import numpy as np
# import csv
# from datetime import datetime
# from PIL import Image
# import torch
# from src.modules.database import preprocess_face
# from facenet_pytorch import MTCNN, InceptionResnetV1
# import faiss  # Install FAISS for fast similarity search

# # Path to the SQLite database
# DATABASE_PATH = r"database/attendance.db"
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# # Initialize MTCNN and FaceNet
# mtcnn = MTCNN(keep_all=True, device='cpu')  # Use GPU if available
# facenet_model = InceptionResnetV1(pretrained="vggface2").eval().cpu()

# # Mark attendance based on group images
# def mark_attendance(filepath, subject, class_no, department, year):
#     """
#     Marks attendance by analyzing the group image and matching detected faces.
#     :param filepath: Path to the uploaded group image.
#     :param subject: Subject for attendance.
#     :param class_no: Class number.
#     :param department: Department of the students.
#     :param year: Academic year.
#     """
#     # Step 1: Load the image and detect faces
#     image = Image.open(filepath).convert("RGB")
#     boxes, _ = mtcnn.detect(image)

#     if boxes is None or len(boxes) == 0:
#         print("No faces detected in the image.")
#         return

#     # Step 2: Preprocess detected faces and generate embeddings
#     detected_faces = [image.crop(tuple(map(int, box))) for box in boxes]
    
#     # Preprocess faces and ensure the correct tensor shape
#     face_tensors = []
#     for face in detected_faces:
#         face_tensor = preprocess_face(face)  # Shape: [1, 3, 160, 160]
#         face_tensors.append(face_tensor.squeeze(0))  # Remove batch dimension, shape: [3, 160, 160]

#     # Stack tensors into a batch: Shape should be [batch_size, 3, 160, 160]
#     face_tensors = torch.stack(face_tensors)

#     with torch.no_grad():
#         detected_embeddings = facenet_model(face_tensors).numpy()

#     # Step 3: Retrieve student embeddings from the database
#     conn = sqlite3.connect(DATABASE_PATH)
#     cursor = conn.cursor()
#     cursor.execute("SELECT prnno, name, embedding FROM students")
#     records = cursor.fetchall()

#     database_embeddings = []
#     student_mapping = []
#     for prnno, name, embedding_blob in records:
#         database_embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))
#         student_mapping.append((prnno, name))

#     # Step 4: Use FAISS for efficient similarity search
#     database_embeddings = np.vstack(database_embeddings)
#     index = faiss.IndexFlatL2(database_embeddings.shape[1])  # L2 distance
#     index.add(database_embeddings)
#     distances, indices = index.search(detected_embeddings, k=1)

#     # Step 5: Match detected embeddings to students
#     attendance_list = []
#     for i, distance in enumerate(distances):
#         if distance[0] < 0.9:  # Matching threshold
#             prnno, name = student_mapping[indices[i][0]]
#             attendance_list.append((prnno, name))

#     # Step 6: Save attendance records to a CSV file
#     csv_dir = "attendance_records"
#     os.makedirs(csv_dir, exist_ok=True)
#     csv_path = os.path.join(csv_dir, f"attendance_{subject}_{class_no}_{department}_{year}.csv")

#     with open(csv_path, mode="a", newline="") as csvfile:
#         writer = csv.writer(csvfile)
#         if os.stat(csv_path).st_size == 0:  # Add header if the file is empty
#             writer.writerow(["PRN No", "Name", "Subject", "Class", "Department", "Year", "Timestamp"])
#         timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
#         for prnno, name in attendance_list:
#             writer.writerow([prnno, name, subject, class_no, department, year, timestamp])

#     print(f"Attendance marked for {len(attendance_list)} students.")


import os
import sqlite3
import numpy as np
import csv
from datetime import datetime
from PIL import Image
import torch
from src.modules.database import preprocess_face
from facenet_pytorch import MTCNN, InceptionResnetV1
import faiss  # Install FAISS for fast similarity search

# Path to the SQLite database
DATABASE_PATH = r"database/attendance.db"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize MTCNN and FaceNet
mtcnn = MTCNN(keep_all=True, device='cpu')  # Use GPU if available
facenet_model = InceptionResnetV1(pretrained="vggface2").eval().cpu()

# Mark attendance based on group images
def mark_attendance(filepath, subject, class_no, department, year):
    """
    Marks attendance by analyzing the group image and matching detected faces.
    :param filepath: Path to the uploaded group image.
    :param subject: Subject for attendance.
    :param class_no: Class number.
    :param department: Department of the students.
    :param year: Academic year.
    """
    # Step 1: Load the image and detect faces
    image = Image.open(filepath).convert("RGB")
    boxes, _ = mtcnn.detect(image)

    if boxes is None or len(boxes) == 0:
        print("No faces detected in the image.")
        return

    # Step 2: Preprocess detected faces and generate embeddings
    detected_faces = [image.crop(tuple(map(int, box))) for box in boxes]
    
    # Preprocess faces and ensure the correct tensor shape
    face_tensors = []
    for face in detected_faces:
        face_tensor = preprocess_face(face)  # Shape: [1, 3, 160, 160]
        face_tensors.append(face_tensor.squeeze(0))  # Remove batch dimension, shape: [3, 160, 160]

    # Stack tensors into a batch: Shape should be [batch_size, 3, 160, 160]
    face_tensors = torch.stack(face_tensors)

    with torch.no_grad():
        detected_embeddings = facenet_model(face_tensors).numpy()

    # Step 3: Retrieve student embeddings from the database
    conn = sqlite3.connect(DATABASE_PATH)
    cursor = conn.cursor()
    cursor.execute("SELECT prnno, name, embedding FROM students")
    records = cursor.fetchall()

    database_embeddings = []
    student_mapping = []
    for prnno, name, embedding_blob in records:
        database_embeddings.append(np.frombuffer(embedding_blob, dtype=np.float32))
        student_mapping.append((prnno, name))

    # Step 4: Use FAISS for efficient similarity search
    database_embeddings = np.vstack(database_embeddings)
    index = faiss.IndexFlatL2(database_embeddings.shape[1])  # L2 distance
    index.add(database_embeddings)
    distances, indices = index.search(detected_embeddings, k=1)

    # Step 5: Match detected embeddings to students
    attendance_list = []
    for i, distance in enumerate(distances):
        if distance[0] < 0.9:  # Matching threshold
            prnno, name = student_mapping[indices[i][0]]
            attendance_list.append((prnno, name))

    # Step 6: Save attendance records to a CSV file (with duplicate prevention)
    csv_dir = "attendance_records"
    os.makedirs(csv_dir, exist_ok=True)
    csv_path = os.path.join(csv_dir, f"attendance_{subject}_{class_no}_{department}_{year}.csv")

    # Load existing records from the CSV file to avoid duplicates
    existing_records = set()
    if os.path.exists(csv_path) and os.stat(csv_path).st_size > 0:
        with open(csv_path, mode="r") as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header
            for row in reader:
                # Create a unique identifier for each record (PRN No + Subject)
                existing_records.add((row[0], subject))  

    with open(csv_path, mode="a", newline="") as csvfile:
        writer = csv.writer(csvfile)
        if os.stat(csv_path).st_size == 0:  # Add header if the file is empty
            writer.writerow(["PRN No", "Name", "Subject", "Class", "Department", "Year", "Timestamp"])
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for prnno, name in attendance_list:
            # Check if the record is already in existing records
            if (prnno, subject) not in existing_records:
                writer.writerow([prnno, name, subject, class_no, department, year, timestamp])
                existing_records.add((prnno, subject))  # Add the new record to the set

    print(f"Attendance marked for {len(attendance_list)} students.")
