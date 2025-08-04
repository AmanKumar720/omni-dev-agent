"""
Omni-Dev Agent Error Handling System
Provides comprehensive error management, logging, and recovery mechanisms.
"""

import logging
import traceback
import functools
from typing import Any, Optional, Callable, Dict, List
from enum import Enum
from dataclasses import dataclass
from datetime import datetime

class ErrorSeverity(Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

@dataclass
class ErrorContext:
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    component: str
    operation: str
    error_type: str
    message: str
    stacktrace: str
    context_data: Dict[str, Any]
    recovery_attempted: bool = False
    recovery_success: bool = False

class ErrorManager:
    def __init__(self, log_file: str = "error_log.txt"):
        self.log_file = log_file
        self.error_history: List[ErrorContext] = []
        self.recovery_strategies: Dict[str, Callable] = {}
        self.logger = self._setup_logger()

    def _setup_logger(self) -> logging.Logger:
        """Set up logging configuration."""
        logger = logging.getLogger("ErrorManager")
        logger.setLevel(logging.ERROR)
        handler = logging.FileHandler(self.log_file)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        return logger

    def register_recovery_strategy(self, error_type: str, strategy: Callable):
        """Register a recovery strategy for a specific error type."""
        self.recovery_strategies[error_type] = strategy

    def handle_error(self, error: Exception, component: str, operation: str, 
                    severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                    context_data: Dict[str, Any] = None) -> ErrorContext:
        """Handle an error with comprehensive logging and recovery attempts."""
        error_context = ErrorContext(
            error_id=f"{component}_{operation}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            timestamp=datetime.now(),
            severity=severity,
            component=component,
            operation=operation,
            error_type=type(error).__name__,
            message=str(error),
            stacktrace=traceback.format_exc(),
            context_data=context_data or {}
        )

        # Log the error
        self.logger.error(f"Error in {component}.{operation}: {error}")
        self.logger.error(f"Context: {context_data}")
        self.logger.error(f"Stacktrace: {error_context.stacktrace}")

        # Attempt recovery if strategy exists
        if error_context.error_type in self.recovery_strategies:
            try:
                self.recovery_strategies[error_context.error_type](error_context)
                error_context.recovery_attempted = True
                error_context.recovery_success = True
                self.logger.info(f"Recovery successful for error {error_context.error_id}")
            except Exception as recovery_error:
                error_context.recovery_attempted = True
                error_context.recovery_success = False
                self.logger.error(f"Recovery failed for error {error_context.error_id}: {recovery_error}")

        # Store error for analysis
        self.error_history.append(error_context)

        # Handle critical errors
        if severity == ErrorSeverity.CRITICAL:
            self._handle_critical_error(error_context)

        return error_context

    def _handle_critical_error(self, error_context: ErrorContext):
        """Handle critical errors that may require system shutdown or alerts."""
        print(f"CRITICAL ERROR: {error_context.message}")
        # In a real system, this might send alerts, create tickets, etc.

    def analyze_error_patterns(self) -> Dict[str, Any]:
        """Analyze error patterns to identify systemic issues."""
        if not self.error_history:
            return {"message": "No errors recorded"}

        error_types = {}
        components_affected = {}
        
        for error in self.error_history:
            # Count error types
            if error.error_type not in error_types:
                error_types[error.error_type] = 0
            error_types[error.error_type] += 1

            # Count components affected
            if error.component not in components_affected:
                components_affected[error.component] = 0
            components_affected[error.component] += 1

        return {
            "total_errors": len(self.error_history),
            "error_types": error_types,
            "components_affected": components_affected,
            "most_common_error": max(error_types, key=error_types.get) if error_types else None,
            "most_affected_component": max(components_affected, key=components_affected.get) if components_affected else None
        }

    def get_error_recommendations(self) -> List[str]:
        """Generate recommendations based on error analysis."""
        analysis = self.analyze_error_patterns()
        recommendations = []

        if analysis.get("most_common_error"):
            recommendations.append(f"Focus on fixing {analysis['most_common_error']} errors")

        if analysis.get("most_affected_component"):
            recommendations.append(f"Review and refactor {analysis['most_affected_component']} component")

        critical_errors = [e for e in self.error_history if e.severity == ErrorSeverity.CRITICAL]
        if critical_errors:
            recommendations.append("Address critical errors immediately to prevent system failures")

        return recommendations

def error_handler(component: str, operation: str, severity: ErrorSeverity = ErrorSeverity.MEDIUM, 
                 error_manager: Optional[ErrorManager] = None):
    """Decorator for automatic error handling."""
    def decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if error_manager:
                    error_manager.handle_error(e, component, operation, severity)
                else:
                    # Fallback logging
                    logging.error(f"Error in {component}.{operation}: {e}")
                raise
        return wrapper
    return decorator

# Global error manager instance
global_error_manager = ErrorManager()

def handle_integration_error(error_context: ErrorContext):
    """Recovery strategy for integration errors."""
    print(f"Attempting to recover from integration error: {error_context.message}")
    # Implement specific recovery logic here

def handle_validation_error(error_context: ErrorContext):
    """Recovery strategy for validation errors."""
    print(f"Attempting to recover from validation error: {error_context.message}")
    # Implement specific recovery logic here

# Register default recovery strategies
global_error_manager.register_recovery_strategy("IntegrationError", handle_integration_error)
global_error_manager.register_recovery_strategy("ValidationError", handle_validation_error)
