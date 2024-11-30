import React from 'react';
import { Card, CardActionArea, CardContent, Typography } from '@mui/material';
import { useNavigate } from 'react-router-dom';
import CardMedia from '@mui/material/CardMedia';

import AddStudent from "../assets/Add_student.jpeg"
import Background from "../assets/WhiteBackground.jpg"
import markAttendance from "../assets/mark_attendance.jpeg"
import Logo from "../assets/Logo.png"; // Import your logo image

const AttendanceSystem = () => {
  const navigate = useNavigate();

  return (
    <div
    style={{
      fontFamily: "Arial, sans-serif",
      display: "flex",
      flexDirection: "column",
      alignItems: "center",
      justifyContent: "center",
      height: "100vh",
      margin: 0,
      backgroundColor: "#f4f4f9",
      backgroundImage: `url(${Background})`, // Use template literals to embed the variable
      backgroundSize: "cover", // Optional: adjust the size of the image
      backgroundPosition: "center", // Optional: align the image
      backgroundRepeat: "no-repeat", // Optional: avoid tiling
    }}
  >
    <div
        style={{
          position: "absolute",
          top: "10px",
          left: "20px",
          display: "flex",
          alignItems: "center",
          gap: "10px",
        }}
      >
        <img
          src={Logo}
          alt="Logo"
          style={{ width: "50px", height: "50px", borderRadius: "50%" }}
        />
        <h2 style={{ margin: 0, color: "#333" }}>VIIT</h2>
      </div>
      <h1 style={{ color: "#333" }}>Attendance System</h1>
      <div
        style={{
          display: "flex",
          gap: "20px",
          marginTop: "20px",
        }}
      >
        {/* Add Student Card */}
        <Card
          sx={{
            width: 300,
            boxShadow: 3,
            borderRadius: "10px",
            transition: "transform 0.3s ease",
            "&:hover": { transform: "scale(1.05)" },
          }}
        >
          <CardActionArea onClick={() => navigate("/add_student")}>
            <CardContent>
              <Typography variant="h5" component="div" color="#4CAF50" gutterBottom>
                Add Student
              </Typography>
              {/* <CardMedia
        component="img"
        height="220" 
        image={AddStudent}
        alt="Paella dish"
      /> */}
              <Typography variant="body2" color="text.secondary">
                Add new students to the attendance system by providing their details.
              </Typography>
            </CardContent>
          </CardActionArea>
        </Card>

        {/* Mark Attendance Card */}
        <Card
          sx={{
            width: 300,
            boxShadow: 3,
            borderRadius: "10px",
            transition: "transform 0.3s ease",
            "&:hover": { transform: "scale(1.05)" },
          }}
        >
          <CardActionArea onClick={() => navigate("/mark_attendance")}>
            <CardContent>
              <Typography variant="h5" component="div" color="#4CAF50" gutterBottom>
                Mark Attendance
              </Typography>
              {/* <CardMedia
        component="img"
        height="220"
        
        image={markAttendance}
        alt="Paella dish"
      /> */}
              <Typography variant="body2" color="text.secondary">
                Mark attendance for the students in your class or session.
              </Typography>
            </CardContent>
          </CardActionArea>
        </Card>
      </div>
    </div>
  );
};

export default AttendanceSystem;
