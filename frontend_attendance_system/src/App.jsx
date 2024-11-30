import { useState } from 'react'

import './App.css'
import Home from './components/Home'
import AddStudent from './components/AddStudent'
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import MarkAttendance from './components/MarkAttendance'

function App() {
  

  return (
    <>
  
      <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/add_student" element={<AddStudent />} />
        <Route path="/mark_attendance" element={<MarkAttendance />} />
      </Routes>
    </Router>
    </>
  )
}

export default App
