"""
Enhanced Crypto Trading Pipeline - Reporting Module

This module handles comprehensive report generation and visualization:
- PDF report generation with charts and analysis
- Email distribution and alerts
- Performance tracking and metrics
- Interactive dashboards and visualizations
"""

from .report_generator import ReportGenerator, DailyReport
from .visualizations import ChartGenerator, DashboardCreator
from .email_sender import EmailSender, AlertManager

__all__ = [
    'ReportGenerator',
    'DailyReport',
    'ChartGenerator',
    'DashboardCreator',
    'EmailSender',
    'AlertManager'
]
