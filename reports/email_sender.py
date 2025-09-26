"""
Email utilities for the crypto trading pipeline
"""

import logging
from typing import Dict, List, Optional, Any
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

logger = logging.getLogger(__name__)

class EmailSender:
    """Basic email sending utilities"""
    
    def __init__(self, smtp_server: str = "smtp.gmail.com", smtp_port: int = 587):
        self.smtp_server = smtp_server
        self.smtp_port = smtp_port
        self.logger = logging.getLogger(__name__)
    
    def send_email(self, to_email: str, subject: str, body: str, 
                   from_email: str = None, password: str = None) -> bool:
        """Send a basic email"""
        try:
            if not from_email or not password:
                self.logger.warning("Email credentials not provided")
                return False
            
            msg = MIMEMultipart()
            msg['From'] = from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            msg.attach(MIMEText(body, 'plain'))
            
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(from_email, password)
            text = msg.as_string()
            server.sendmail(from_email, to_email, text)
            server.quit()
            
            self.logger.info(f"Email sent to {to_email}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error sending email: {e}")
            return False

class AlertManager:
    """Basic alert management utilities"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.alerts = []
    
    def add_alert(self, message: str, level: str = "INFO") -> None:
        """Add an alert message"""
        alert = {
            'message': message,
            'level': level,
            'timestamp': None  # Would be datetime.now() in real implementation
        }
        self.alerts.append(alert)
        self.logger.info(f"Alert added: {message}")
    
    def get_alerts(self, level: str = None) -> List[Dict[str, Any]]:
        """Get alerts, optionally filtered by level"""
        if level:
            return [alert for alert in self.alerts if alert['level'] == level]
        return self.alerts
    
    def clear_alerts(self) -> None:
        """Clear all alerts"""
        self.alerts.clear()
        self.logger.info("All alerts cleared")
