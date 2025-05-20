import React from 'react';
import axios from 'axios';

const DownloadAttendanceCSV = () => {
    const handleDownload = async () => {
        try {
            const response = await axios.get('http://localhost:5000/download_csv', {
                responseType: 'blob', // Ensure the response is treated as a file
            });

            // Create a download URL for the received file
            const url = window.URL.createObjectURL(new Blob([response.data]));
            const link = document.createElement('a');
            link.href = url;
            link.setAttribute('download', 'attendance.csv'); // File name for download
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link); // Cleanup
        } catch (error) {
            console.error('Error downloading CSV:', error);
            alert('Failed to download attendance CSV. Please try again.');
        }
    };

    return (
        <div>
            <button
                onClick={handleDownload}
                className="bg-blue-500 hover:bg-blue-700 text-white font-bold py-2 px-4 rounded"
            >
                Download Attendance CSV
            </button>
        </div>
    );
};

export default DownloadAttendanceCSV;
