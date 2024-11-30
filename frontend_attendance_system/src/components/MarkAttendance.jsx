import React, { useState } from "react";
import {
  Box,
  Button,
  TextField,
  Typography,
  Card,
  CardContent,
  InputLabel,
  Alert,
  AlertTitle,
} from "@mui/material";

const MarkAttendance = () => {
  const [formData, setFormData] = useState({
    subject: "",
    classNo: "",
    department: "",
    year: "",
    image: null,
  });

  const [alert, setAlert] = useState(null); // State for displaying alerts

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prevData) => ({
      ...prevData,
      [name]: value,
    }));
  };

  const handleFileChange = (e) => {
    setFormData((prevData) => ({
      ...prevData,
      image: e.target.files[0],
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    const data = new FormData();
    data.append("subject", formData.subject);
    data.append("class_no", formData.classNo);
    data.append("department", formData.department);
    data.append("year", formData.year);
    data.append("image", formData.image);

    try {
      const response = await fetch("http://127.0.0.1:5000/upload", {
        method: "POST",
        body: data,
      });

      if (response.ok) {
        setAlert({
          type: "success",
          title: "Success",
          message: "Upload successful!",
        });
        setFormData({
          subject: "",
          classNo: "",
          department: "",
          year: "",
          image: null,
        });
      } else {
        setAlert({
          type: "error",
          title: "Error",
          message: "Failed to upload. Please try again.",
        });
      }
    } catch (error) {
      console.error("Error uploading:", error);
      setAlert({
        type: "error",
        title: "Error",
        message: "An error occurred during the upload.",
      });
    }
  };

  return (
    <Box
      sx={{
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        minHeight: "100vh",
        backgroundColor: "#f4f4f9",
        padding: 2,
      }}
    >
      <Card sx={{ maxWidth: 400, width: "100%", boxShadow: 3 }}>
        <CardContent>
          <Typography variant="h5" gutterBottom align="center">
            Upload Group Image for Attendance
          </Typography>

          {alert && (
            <Alert severity={alert.type} sx={{ mb: 2 }}>
              <AlertTitle>{alert.title}</AlertTitle>
              {alert.message}
            </Alert>
          )}

          <form onSubmit={handleSubmit}>
            <TextField
              label="Subject Name"
              name="subject"
              value={formData.subject}
              onChange={handleChange}
              fullWidth
              required
              margin="normal"
              variant="outlined"
            />
            <TextField
              label="Class No"
              name="classNo"
              value={formData.classNo}
              onChange={handleChange}
              fullWidth
              required
              margin="normal"
              variant="outlined"
            />
            <TextField
              label="Department Name"
              name="department"
              value={formData.department}
              onChange={handleChange}
              fullWidth
              required
              margin="normal"
              variant="outlined"
            />
            <TextField
              label="Year of Study"
              name="year"
              value={formData.year}
              onChange={handleChange}
              fullWidth
              required
              margin="normal"
              variant="outlined"
            />
            <InputLabel htmlFor="image" sx={{ mt: 2, mb: 1 }}>
              Upload Group Image
            </InputLabel>
            <input
              type="file"
              id="image"
              name="image"
              accept="image/*"
              onChange={handleFileChange}
              style={{
                display: "block",
                marginBottom: "20px",
                width: "100%",
              }}
              required
            />
            <Button
              type="submit"
              variant="contained"
              color="primary"
              fullWidth
              sx={{ mt: 2 }}
            >
              Upload
            </Button>
          </form>
        </CardContent>
      </Card>
    </Box>
  );
};

export default MarkAttendance;
