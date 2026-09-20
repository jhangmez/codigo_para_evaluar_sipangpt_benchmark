"""
Módulo de generación de informes, gráficos y exportaciones para la tesis.
"""

from sipangpt_eval.reporting.charts import generate_all_reports_and_charts
from sipangpt_eval.reporting.excel_generator import (
    generate_comparative_excel,
    generate_jev_comparative_excel,
)

__all__ = [
    "generate_comparative_excel",
    "generate_jev_comparative_excel",
    "generate_all_reports_and_charts",
]
