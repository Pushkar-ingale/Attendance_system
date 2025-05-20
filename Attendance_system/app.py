from flask import Flask, render_template, request, jsonify, redirect, url_for, send_file
from werkzeug.utils import secure_filename
from flask_cors import CORS  # Import CORS
from src.modules.db_helper import initialize_database
from src.modules.database import capture_faces
from src.modules.Attendance_model import mark_attendance
import logging
import os
import traceback

# Initialize Flask app
app = Flask(__name__)

# Enable CORS for the entire app
CORS(app)
# Configuration
UPLOAD_FOLDER = 'uploads'
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # Limit upload size to 5 MB

# Ensure the upload directory exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Logging configuration
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

# Initialize the database
initialize_database()

# Helper functions
def allowed_file(filename):
    """Check if the file has an allowed extension."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def cleanup_file(filepath):
    """Delete a file if it exists."""
    try:
        if os.path.exists(filepath):
            os.remove(filepath)
            logging.info(f"File {filepath} cleaned up successfully.")
    except Exception as e:
        logging.error(f"Error during cleanup: {e}")


@app.route('/submit', methods=['POST'])
def submit():
    """Endpoint to add student data."""
    try:
        data = request.get_json()

        # Validate request data
        prnno = data.get('prnno')
        name = data.get('name')
        images = data.get('images')

        if not prnno or not name or not images:
            return jsonify({"error": "Missing required fields"}), 400

        # Capture student faces
        capture_faces(prnno, name, images)

        return jsonify({"message": "Student data submitted successfully!"}), 200
    except ValueError as e:
        logging.error(f"ValueError in /submit: {e}")
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logging.error(f"Unexpected error in /submit: {traceback.format_exc()}")
        return jsonify({"error": "An unexpected error occurred."}), 500

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle the file upload and related form data."""
    try:
        # Retrieve form data
        subject = request.form.get('subject')
        class_no = request.form.get('class_no')
        department = request.form.get('department')
        year = request.form.get('year')

        # Validate form data
        if not subject or not class_no or not department or not year:
            return jsonify({"error": "Missing form data"}), 400

        # Retrieve and validate the uploaded file
        file = request.files.get('image')
        if not file or file.filename == '':
            return jsonify({"error": "No file uploaded"}), 400
        if not allowed_file(file.filename):
            return jsonify({"error": "Invalid file type. Allowed: png, jpg, jpeg"}), 400

        # Save the uploaded file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        logging.info(f"File uploaded: {filepath}")

        # Redirect to process attendance with required params
        return redirect(url_for('process_attendance', filepath=filepath, subject=subject,
                                class_no=class_no, department=department, year=year))
    except Exception as e:
        logging.error(f"Error in /upload: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred while processing your request."}), 500

@app.route('/process_attendance', methods=['GET'])
def process_attendance():
    try:
        # Extract parameters from the request
        filepath = request.args.get('filepath')
        subject = request.args.get('subject')
        class_no = request.args.get('class_no')
        department = request.args.get('department')
        year = request.args.get('year')

        # Check if any of the required parameters are missing
        if not filepath or not subject or not class_no or not department or not year:
            raise ValueError("Missing required parameter(s).")

        # Call the mark_attendance function and get the filename
        csv_filename = mark_attendance(filepath, subject, class_no, department, year)
        cleanup_file(filepath)

        return jsonify({
            "message": "Attendance marked successfully!",
            "csv_filename": csv_filename  # This will be used to download the file
        }), 200
    except Exception as e:
        logging.error(f"Error in /process_attendance: {traceback.format_exc()}")
        return jsonify({"error": "An error occurred while processing attendance."}), 500

csv_dir = "attendance_records"  # Directory where CSVs are stored
@app.route('/download_csv', methods=['GET'])
def download_csv():
    filename = request.args.get('filename')
    print(filename)
    # Check if filename is provided and valid
    if not filename or filename == 'null':
        return jsonify({"error": "Invalid filename"}), 400

    # Construct the full path
    filepath = os.path.join(csv_dir, filename)
    print(filepath)
    # Check if the file exists
    if not os.path.isfile(filepath):
        return jsonify({"error": "File not found"}), 404

    return send_file(filepath, as_attachment=True, download_name=filename)

# Main function
if __name__ == '__main__':
    app.run(debug=True)
