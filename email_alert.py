import smtplib, os
from email.message import EmailMessage
from dotenv import load_dotenv

# Load in environment variables
load_dotenv()

# SMTP Configuration
SMTP_SERVER = "smtp.gmail.com"
SMTP_PORT = 587
SMTP_USER = os.environ.get("SMTP_EMAIL")    # Your SMTP email from environment variable
SMTP_PASS = os.environ.get("SMTP_PASSWORD") # Your SMTP password from environment variable
SUPERVISOR_EMAIL = "lolsal4d@gmail.com" # Enter recipient email address here

def send_alert(image_path):
    msg = EmailMessage()
    msg["Subject"] = "PPE Violation Detected"
    msg["From"] = SMTP_USER
    msg["To"] = SUPERVISOR_EMAIL
    msg.set_content("A PPE non-compliance has been detected. The attached image shows the violation as evidence.")

    with open(image_path, "rb") as f:
        msg.add_attachment(
            f.read(),
            maintype="image",
            subtype="jpeg",
            filename="violation.jpg"
        )

    with smtplib.SMTP(SMTP_SERVER, SMTP_PORT) as server:
        server.starttls()
        server.login(SMTP_USER, SMTP_PASS)
        server.send_message(msg)
