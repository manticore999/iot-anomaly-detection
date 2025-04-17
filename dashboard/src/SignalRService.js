import { HubConnectionBuilder } from '@microsoft/signalr';

export const connectToSignalR = (onAlertReceived) => {
  const connection = new HubConnectionBuilder()
    .withUrl('http://localhost:7071/api')
    .withAutomaticReconnect()
    .build();

  connection.on('ReceiveAlert', onAlertReceived);

  connection.start()
    .then(() => console.log('SignalR Connected'))
    .catch(err => console.error('SignalR Connection Error:', err));

  return connection;
};