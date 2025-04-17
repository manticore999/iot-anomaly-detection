import azure.functions as func
import json
import requests
import os
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv


load_dotenv()

app = func.FunctionApp()

def send_email_alert(alert_message):
    """Send email alert when anomalies are detected using environment variables"""
    
    sender_email = os.getenv("SENDER_EMAIL")
    receiver_email = os.getenv("RECEIVER_EMAIL")
    smtp_server = os.getenv("SMTP_SERVER")
    smtp_port = int(os.getenv("SMTP_PORT", 587))
    smtp_username = os.getenv("SMTP_USERNAME")
    smtp_password = os.getenv("SMTP_PASSWORD")

   
    if not all([sender_email, receiver_email, smtp_server, smtp_username, smtp_password]):
        raise ValueError("Missing required email configuration in environment variables")

    
    message = MIMEMultipart()
    message["From"] = sender_email
    message["To"] = receiver_email
    message["Subject"] = "Temperature Anomaly Alert"
    
    # Email body
    body = f"""
    <h2>Anomaly Alert Notification</h2>
    <p>{alert_message}</p>
    <p>Please review the temperature data for potential issues.</p>
    """
    message.attach(MIMEText(body, "html"))

    try:
        # Send email
        with smtplib.SMTP(smtp_server, smtp_port) as server:
            server.starttls()
            server.login(smtp_username, smtp_password)
            server.sendmail(sender_email, receiver_email, message.as_string())
        return True
    except Exception as e:
        print(f"Failed to send email: {str(e)}")
        return False

@app.function_name(name="AnomalyAlert")
@app.route(route="anomaly_alert", auth_level=func.AuthLevel.ANONYMOUS)
def anomaly_alert(req: func.HttpRequest) -> func.HttpResponse:
    try:
        
        req_body = req.get_json()
        temperature_data = req_body.get("data", [25.0, 26.5, 30.0, 40.0])
        
        
        endpoint_url = os.getenv("ENDPOINT_URL")
        if not endpoint_url:
            return func.HttpResponse("ENDPOINT_URL not configured", status_code=500)
            
        headers = {"Content-Type": "application/json"}
        payload = json.dumps({"data": temperature_data})
        response = requests.post(endpoint_url, data=payload, headers=headers)
        response.raise_for_status()
        result = response.json()
        
        
        threshold = float(os.getenv("ANOMALY_THRESHOLD", 0.01))
        errors = result.get("reconstruction_error", [])
        anomalies = [i for i, error in enumerate(errors) if error > threshold]
        
        if anomalies:
            alert_msg = (
                f"Anomalies detected at indices {anomalies}: "
                f"Errors = {[errors[i] for i in anomalies]}\n"
                f"Temperature Data: {temperature_data}"
            )
            
            
            email_sent = send_email_alert(alert_msg)
            if email_sent:
                return func.HttpResponse(f"Alert sent to {os.getenv('RECEIVER_EMAIL')}: {alert_msg}", status_code=200)
            else:
                return func.HttpResponse(f"Anomaly detected but email failed: {alert_msg}", status_code=200)
        else:
            return func.HttpResponse("No anomalies detected", status_code=200)
            
    except json.JSONDecodeError:
        return func.HttpResponse("Invalid JSON input", status_code=400)
    except requests.RequestException as e:
        return func.HttpResponse(f"Endpoint request failed: {str(e)}", status_code=502)
    except Exception as e:
        return func.HttpResponse(f"Error: {str(e)}", status_code=500)