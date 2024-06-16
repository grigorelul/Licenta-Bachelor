// src/components/ExampleComponent.tsx
import React, { useEffect, useState } from 'react';
import api from '../api/api';

interface Data {
  id: number;
  name: string;
}

const Component: React.FC = () => {
  const [data, setData] = useState<Data[]>([]);

  useEffect(() => {
    const fetchData = async () => {
      try {
        const response = await api.get<Data[]>('/endpoint');
        setData(response.data);
      } catch (error) {
        console.error('Eroare la preluarea datelor', error);
      }
    };

    fetchData();
  }, []);

  return (
    <div>
      <h1>Example Data</h1>
      <ul>
        {data.map(item => (
          <li key={item.id}>{item.name}</li>
        ))}
      </ul>
    </div>
  );
};

export default Component;
