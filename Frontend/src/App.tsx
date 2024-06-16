import React, { useState } from 'react';
import { BrowserRouter as Router, Route, Routes, useNavigate } from 'react-router-dom';
import axios from 'axios';

const Login: React.FC = () => {
  const [email, setEmail] = useState('');
  const [parola, setParola] = useState('');
  const [error, setError] = useState('');
  const navigate = useNavigate();

  const handleLogin = async () => {
    try {
      const response = await axios.post('https://localhost:7057/api/Users/login', { email, parola });
      console.log('Login successful:', response.data);
      localStorage.setItem('user', JSON.stringify(response.data));
      navigate('/home');
    } catch (error) {
      console.error('Error during login:', error);
      setError('Login failed. Please check your email and password.');
    }
  };

  return (
    <div className="login-container">
      <h1>Login</h1>
      <div className="login-form">
        <div className="form-group">
          <label htmlFor="email">Email:</label>
          <input
            type="email"
            id="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
        </div>
        <div className="form-group">
          <label htmlFor="parola">Parola:</label>
          <input
            type="password"
            id="parola"
            value={parola}
            onChange={(e) => setParola(e.target.value)}
            required
          />
        </div>
        {error && <div className="error">{error}</div>}
        <button onClick={handleLogin}>Login</button>
      </div>
    </div>
  );
};

const Home: React.FC = () => {
  return (
    <div>
      <h1>Home Page</h1>
      <p>Welcome to the Home Page!</p>
    </div>
  );
};

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Login />} />
        <Route path="/home" element={<Home />} />
      </Routes>
    </Router>
  );
};

export default App;
