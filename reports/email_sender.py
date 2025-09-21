"""
Email Sending and Alert Management

Sends reports and alerts via email
"""

import smtplib
import ssl
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from email.mime.base import MIMEBase
from email import encoders
import pandas as pd
from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from loguru import logger
import os
from datetime import datetime


@dataclass
class EmailConfig:
    """Email configuration"""
    smtp_server: str
    smtp_port: int
    username: str
    password: str
    from_email: str
    to_emails: List[str]


class EmailSender:
    """Send emails with reports and alerts"""
    
    def __init__(self, config: EmailConfig):
        self.config = config
    
    def send_report_email(self, subject: str, body: str, attachments: List[str] = None) -> bool:
        """Send email with report"""
        try:
            msg = MIMEMultipart()
            msg['From'] = self.config.from_email
            msg['To'] = ', '.join(self.config.to_emails)
            msg['Subject'] = subject
            
            # Add body
            msg.attach(MIMEText(body, 'html'))
            
            # Add attachments
            if attachments:
                for file_path in attachments:
                    if os.path.exists(file_path):
                        with open(file_path, "rb") as attachment:
                            part = MIMEBase('application', 'octet-stream')
                            part.set_payload(attachment.read())
                            encoders.encode_base64(part)
                            part.add_header(
                                'Content-Disposition',
                                f'attachment; filename= {os.path.basename(file_path)}'
                            )
                            msg.attach(part)
            
            # Send email
            context = ssl.create_default_context()
            with smtplib.SMTP(self.config.smtp_server, self.config.smtp_port) as server:
                server.starttls(context=context)
                server.login(self.config.username, self.config.password)
                server.send_message(msg)
            
            logger.info(f"Report email sent successfully to {len(self.config.to_emails)} recipients")
            return True
            
        except Exception as e:
            logger.error(f"Error sending email: {e}")
            return False
    
    def send_alert_email(self, alert_type: str, message: str) -> bool:
        """Send alert email"""
        subject = f"Trading Alert: {alert_type}"
        body = f"""
        <h2>Trading Alert</h2>
        <p><strong>Type:</strong> {alert_type}</p>
        <p><strong>Message:</strong> {message}</p>
        <p><strong>Time:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        """
        
        return self.send_report_email(subject, body)


class AlertManager:
    """Manage trading alerts and notifications"""
    
    def __init__(self, email_sender: EmailSender):
        self.email_sender = email_sender
        self.alert_history = []
    
    def create_alert(self, alert_type: str, message: str, priority: str = 'medium') -> bool:
        """Create and send alert"""
        alert = {
            'type': alert_type,
            'message': message,
            'priority': priority,
            'timestamp': datetime.now(),
            'sent': False
        }
        
        # Send email alert
        if self.email_sender:
            alert['sent'] = self.email_sender.send_alert_email(alert_type, message)
        
        self.alert_history.append(alert)
        
        if alert['sent']:
            logger.info(f"Alert sent: {alert_type} - {message}")
        else:
            logger.error(f"Failed to send alert: {alert_type} - {message}")
        
        return alert['sent']
    
    def get_alert_history(self, hours_back: int = 24) -> List[Dict]:
        """Get recent alert history"""
        cutoff_time = datetime.now() - timedelta(hours=hours_back)
        return [alert for alert in self.alert_history if alert['timestamp'] >= cutoff_time]

