// src/api/api.ts
import axios from 'axios';

const api = axios.create({
  baseURL: 'https://api.example.com', // înlocuiește cu URL-ul real al API-ului tău
  headers: {
    'Content-Type': 'application/json',
  },
});

export default api;