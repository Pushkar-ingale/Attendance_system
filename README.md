# 📸 Attendance System Using Face Recognition

This project is a **face recognition-based attendance system** designed to simplify and automate the attendance-taking process. The system leverages **Flask**, **FaceNet**, **MTCNN**, and a combination of **Python**, **HTML**, **SQLite**, and **CSV** to provide a seamless experience for both students and teachers.

---

## 🎯 Features
1. **Student Registration**:
   - Students register themselves in the database through a user-friendly interface.
   - The system captures **30 images** of the student during registration.
   - Converts the images into **embeddings** using FaceNet and creates a **super embedding** for accurate recognition.

2. **Attendance Marking**:
   - Teachers upload a single image of the class along with session details.
   - The system identifies all students present in the image.
   - Attendance is automatically marked in a **CSV file**.

3. **Database Management**:
   - All student data is securely stored in an **SQLite database**.
   - Attendance records are exported to CSV for easy reporting and integration with other systems.

---

## 🛠️ Technologies Used
- **Python**: Core programming language for backend logic.
- **Flask**: Framework for creating the web interface and API endpoints.
- **FaceNet**: For generating high-accuracy facial embeddings.
- **MTCNN**: For efficient face detection in images.
- **SQLite**: Lightweight database for storing student records.
- **CSV**: For maintaining and exporting attendance records.
- **HTML/CSS**: Frontend for user interaction and registration.

---

