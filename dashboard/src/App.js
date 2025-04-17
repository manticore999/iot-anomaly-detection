import React, { useState, useEffect } from 'react';
import './App.css';
import { HubConnectionBuilder } from '@microsoft/signalr';

function App() {
  const [data, setData] = useState([]);
  const [alerts, setAlerts] = useState([]);

  // SignalR connection
  useEffect(() => {
    const connection = new HubConnectionBuilder()
      .withUrl('http://localhost:7071/api')  // SignalR endpoint
      .withAutomaticReconnect()
      .build();

    connection.on('ReceiveAlert', (message) => {
      setAlerts((prev) => [...prev, message]);
    });

    connection.start()
      .then(() => console.log('SignalR Connected'))
      .catch(err => console.error('SignalR Connection Error:', err));

    return () => {
      connection.stop();
    };
  }, []);

  
  useEffect(() => {
    
    const fetchData = async () => {
      try {
        const response = await fetch('http://localhost:7071/api/anomaly_alert');
        const result = await response.text();
        setData([
          { temperature: 25.0, reconstructed: 24.98, error: 0.0004 },
          { temperature: 26.5, reconstructed: 26.47, error: 0.0009 },
          { temperature: 30.0, reconstructed: 29.95, error: 0.0025 },
          { temperature: 40.0, reconstructed: 39.80, error: 0.0400 }
        ]);
        if (result.includes('Anomalies detected')) {
          setAlerts((prev) => [...prev, result]);
        }
      } catch (error) {
        console.error('Error fetching data:', error);
      }
    };
    fetchData();
  }, []);

  return (
    <div className="App">
      <h1>IoT Anomaly Detection Dashboard</h1>
      
      <h2>Temperature Data</h2>
      <table>
        <thead>
          <tr>
            <th>Temperature (°C)</th>
            <th>Reconstructed (°C)</th>
            <th>Reconstruction Error</th>
            <th>Status</th>
          </tr>
        </thead>
        <tbody>
          {data.map((row, index) => (
            <tr key={index} className={row.error > 0.01 ? 'anomaly' : ''}>
              <td>{row.temperature}</td>
              <td>{row.reconstructed}</td>
              <td>{row.error.toFixed(4)}</td>
              <td>{row.error > 0.01 ? 'Anomaly' : 'Normal'}</td>
            </tr>
          ))}
        </tbody>
      </table>

      <h2>Alerts</h2>
      <ul>
        {alerts.map((alert, index) => (
          <li key={index} className="alert">{alert}</li>
        ))}
      </ul>
    </div>
  );
}

export default App;