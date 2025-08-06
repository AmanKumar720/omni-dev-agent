import os
import yaml
import json
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field, asdict
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


@dataclass
class ThreadedWorkerConfig:
    """Configuration for threaded background worker."""
    
    max_workers: int = 4
    queue_size: int = 100
    
    def validate(self):
        """Validate configuration values."""
        if self.max_workers < 1:
            raise ValueError("max_workers must be at least 1")
        if self.queue_size < 1:
            raise ValueError("queue_size must be at least 1")


@dataclass
class CeleryWorkerConfig:
    """Configuration for Celery background worker."""
    
    broker_url: str = "redis://localhost:6379/0"
    result_backend: str = "redis://localhost:6379/0"
    task_serializer: str = "json"
    accept_content: List[str] = field(default_factory=lambda: ["json"])
    result_serializer: str = "json"
    timezone: str = "UTC"
    enable_utc: bool = True
    worker_prefetch_multiplier: int = 1
    task_acks_late: bool = True
    worker_max_tasks_per_child: int = 1000
    result_expires: int = 3600
    
    # Queue configuration
    default_queue: str = "omni_dev"
    priority_queue: str = "priority"
    scheduled_queue: str = "scheduled"
    
    # Beat scheduler configuration
    beat_schedule_file: Optional[str] = None
    
    def validate(self):
        """Validate configuration values."""
        if not self.broker_url:
            raise ValueError("broker_url is required")
        if not self.result_backend:
            raise ValueError("result_backend is required")
        if self.worker_prefetch_multiplier < 1:
            raise ValueError("worker_prefetch_multiplier must be at least 1")
        if self.result_expires < 0:
            raise ValueError("result_expires must be non-negative")
    
    @classmethod
    def from_env(cls) -> 'CeleryWorkerConfig':
        """Create configuration from environment variables."""
        return cls(
            broker_url=os.getenv('CELERY_BROKER_URL', cls.broker_url),
            result_backend=os.getenv('CELERY_RESULT_BACKEND', cls.result_backend),
            task_serializer=os.getenv('CELERY_TASK_SERIALIZER', cls.task_serializer),
            result_serializer=os.getenv('CELERY_RESULT_SERIALIZER', cls.result_serializer),
            timezone=os.getenv('CELERY_TIMEZONE', cls.timezone),
            enable_utc=os.getenv('CELERY_ENABLE_UTC', str(cls.enable_utc)).lower() == 'true',
            worker_prefetch_multiplier=int(os.getenv('CELERY_WORKER_PREFETCH_MULTIPLIER', str(cls.worker_prefetch_multiplier))),
            task_acks_late=os.getenv('CELERY_TASK_ACKS_LATE', str(cls.task_acks_late)).lower() == 'true',
            worker_max_tasks_per_child=int(os.getenv('CELERY_WORKER_MAX_TASKS_PER_CHILD', str(cls.worker_max_tasks_per_child))),
            result_expires=int(os.getenv('CELERY_RESULT_EXPIRES', str(cls.result_expires))),
            default_queue=os.getenv('CELERY_DEFAULT_QUEUE', cls.default_queue),
            priority_queue=os.getenv('CELERY_PRIORITY_QUEUE', cls.priority_queue),
            scheduled_queue=os.getenv('CELERY_SCHEDULED_QUEUE', cls.scheduled_queue),
        )


@dataclass
class MonitoringConfig:
    """Configuration for task monitoring."""
    
    enabled: bool = True
    max_history_size: int = 1000
    alert_enabled: bool = True
    
    # Alert thresholds
    error_rate_threshold: float = 0.1  # 10%
    avg_execution_time_threshold: int = 300  # 5 minutes
    queue_size_threshold: int = 100
    
    # Metrics export
    export_enabled: bool = False
    export_interval_seconds: int = 3600  # 1 hour
    export_format: str = "json"
    export_path: str = "/tmp/omni_dev_metrics"
    
    def validate(self):
        """Validate configuration values."""
        if self.max_history_size < 1:
            raise ValueError("max_history_size must be at least 1")
        if not 0 <= self.error_rate_threshold <= 1:
            raise ValueError("error_rate_threshold must be between 0 and 1")
        if self.avg_execution_time_threshold < 0:
            raise ValueError("avg_execution_time_threshold must be non-negative")
        if self.queue_size_threshold < 0:
            raise ValueError("queue_size_threshold must be non-negative")


@dataclass
class SchedulerConfig:
    """Configuration for task scheduler."""
    
    enabled: bool = True
    check_interval_seconds: float = 1.0
    timezone: str = "UTC"
    
    def validate(self):
        """Validate configuration values."""
        if self.check_interval_seconds <= 0:
            raise ValueError("check_interval_seconds must be positive")


@dataclass
class BackgroundTaskConfig:
    """Main configuration for background task system."""
    
    # Worker configuration
    worker_type: str = "threaded"  # "threaded" or "celery"
    threaded_worker: ThreadedWorkerConfig = field(default_factory=ThreadedWorkerConfig)
    celery_worker: CeleryWorkerConfig = field(default_factory=CeleryWorkerConfig)
    
    # Monitoring configuration
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    
    # Scheduler configuration
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)
    
    # Global settings
    log_level: str = "INFO"
    shutdown_timeout_seconds: int = 30
    
    def validate(self):
        """Validate the entire configuration."""
        if self.worker_type not in ["threaded", "celery"]:
            raise ValueError("worker_type must be 'threaded' or 'celery'")
        
        # Validate log level
        valid_log_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if self.log_level.upper() not in valid_log_levels:
            raise ValueError(f"log_level must be one of {valid_log_levels}")
        
        if self.shutdown_timeout_seconds < 0:
            raise ValueError("shutdown_timeout_seconds must be non-negative")
        
        # Validate sub-configurations
        self.threaded_worker.validate()
        self.celery_worker.validate()
        self.monitoring.validate()
        self.scheduler.validate()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'BackgroundTaskConfig':
        """Create configuration from dictionary."""
        # Handle nested configurations
        if 'threaded_worker' in data:
            data['threaded_worker'] = ThreadedWorkerConfig(**data['threaded_worker'])
        
        if 'celery_worker' in data:
            data['celery_worker'] = CeleryWorkerConfig(**data['celery_worker'])
        
        if 'monitoring' in data:
            data['monitoring'] = MonitoringConfig(**data['monitoring'])
        
        if 'scheduler' in data:
            data['scheduler'] = SchedulerConfig(**data['scheduler'])
        
        return cls(**data)
    
    @classmethod
    def from_file(cls, config_path: str) -> 'BackgroundTaskConfig':
        """Load configuration from file (JSON or YAML)."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            elif config_path.suffix.lower() == '.json':
                data = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_env(cls) -> 'BackgroundTaskConfig':
        """Create configuration from environment variables."""
        config = cls()
        
        # Worker type
        config.worker_type = os.getenv('BG_WORKER_TYPE', config.worker_type)
        
        # Threaded worker settings
        if os.getenv('BG_THREADED_MAX_WORKERS'):
            config.threaded_worker.max_workers = int(os.getenv('BG_THREADED_MAX_WORKERS'))
        if os.getenv('BG_THREADED_QUEUE_SIZE'):
            config.threaded_worker.queue_size = int(os.getenv('BG_THREADED_QUEUE_SIZE'))
        
        # Celery worker settings (use CeleryWorkerConfig.from_env)
        config.celery_worker = CeleryWorkerConfig.from_env()
        
        # Monitoring settings
        if os.getenv('BG_MONITORING_ENABLED'):
            config.monitoring.enabled = os.getenv('BG_MONITORING_ENABLED').lower() == 'true'
        if os.getenv('BG_MONITORING_MAX_HISTORY'):
            config.monitoring.max_history_size = int(os.getenv('BG_MONITORING_MAX_HISTORY'))
        if os.getenv('BG_MONITORING_ERROR_THRESHOLD'):
            config.monitoring.error_rate_threshold = float(os.getenv('BG_MONITORING_ERROR_THRESHOLD'))
        
        # Scheduler settings
        if os.getenv('BG_SCHEDULER_ENABLED'):
            config.scheduler.enabled = os.getenv('BG_SCHEDULER_ENABLED').lower() == 'true'
        if os.getenv('BG_SCHEDULER_CHECK_INTERVAL'):
            config.scheduler.check_interval_seconds = float(os.getenv('BG_SCHEDULER_CHECK_INTERVAL'))
        
        # Global settings
        config.log_level = os.getenv('BG_LOG_LEVEL', config.log_level)
        if os.getenv('BG_SHUTDOWN_TIMEOUT'):
            config.shutdown_timeout_seconds = int(os.getenv('BG_SHUTDOWN_TIMEOUT'))
        
        return config
    
    def save_to_file(self, config_path: str):
        """Save configuration to file."""
        config_path = Path(config_path)
        
        # Create directory if it doesn't exist
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        data = self.to_dict()
        
        with open(config_path, 'w') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                yaml.safe_dump(data, f, indent=2)
            elif config_path.suffix.lower() == '.json':
                json.dump(data, f, indent=2, default=str)
            else:
                raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
        
        logger.info(f"Configuration saved to {config_path}")


class ConfigManager:
    """Manager for background task configuration."""
    
    def __init__(self, config: Optional[BackgroundTaskConfig] = None):
        self.config = config or BackgroundTaskConfig()
        self._validated = False
    
    def load_from_file(self, config_path: str):
        """Load configuration from file."""
        self.config = BackgroundTaskConfig.from_file(config_path)
        self._validated = False
        logger.info(f"Configuration loaded from {config_path}")
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        self.config = BackgroundTaskConfig.from_env()
        self._validated = False
        logger.info("Configuration loaded from environment variables")
    
    def load_from_dict(self, data: Dict[str, Any]):
        """Load configuration from dictionary."""
        self.config = BackgroundTaskConfig.from_dict(data)
        self._validated = False
        logger.info("Configuration loaded from dictionary")
    
    def validate(self):
        """Validate the current configuration."""
        try:
            self.config.validate()
            self._validated = True
            logger.info("Configuration validation successful")
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            raise
    
    def get_worker_config(self) -> Dict[str, Any]:
        """Get worker-specific configuration."""
        if not self._validated:
            self.validate()
        
        if self.config.worker_type == "threaded":
            return asdict(self.config.threaded_worker)
        elif self.config.worker_type == "celery":
            return asdict(self.config.celery_worker)
        else:
            raise ValueError(f"Unknown worker type: {self.config.worker_type}")
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration."""
        if not self._validated:
            self.validate()
        
        return asdict(self.config.monitoring)
    
    def get_scheduler_config(self) -> Dict[str, Any]:
        """Get scheduler configuration."""
        if not self._validated:
            self.validate()
        
        return asdict(self.config.scheduler)
    
    def apply_logging_config(self):
        """Apply logging configuration."""
        if not self._validated:
            self.validate()
        
        # Set log level for background task loggers
        log_level = getattr(logging, self.config.log_level.upper())
        
        loggers_to_configure = [
            'src.core.background_worker',
            'src.core.celery_worker', 
            'src.core.task_monitor'
        ]
        
        for logger_name in loggers_to_configure:
            logger = logging.getLogger(logger_name)
            logger.setLevel(log_level)
    
    def save_to_file(self, config_path: str):
        """Save current configuration to file."""
        if not self._validated:
            self.validate()
        
        self.config.save_to_file(config_path)


# Global config manager instance
_config_manager = None


def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager


def load_config(config_source: Optional[str] = None) -> BackgroundTaskConfig:
    """
    Load configuration from various sources.
    
    Args:
        config_source: Path to config file, 'env' for environment variables,
                      or None for default configuration
    
    Returns:
        BackgroundTaskConfig: Loaded and validated configuration
    """
    manager = get_config_manager()
    
    if config_source is None:
        # Use default configuration
        pass
    elif config_source == 'env':
        manager.load_from_env()
    elif isinstance(config_source, str) and Path(config_source).exists():
        manager.load_from_file(config_source)
    else:
        raise ValueError(f"Invalid config source: {config_source}")
    
    manager.validate()
    manager.apply_logging_config()
    
    return manager.config


# Default configuration paths
DEFAULT_CONFIG_PATHS = [
    'omni_dev_bg_config.yaml',
    'omni_dev_bg_config.yml', 
    'omni_dev_bg_config.json',
    '/etc/omni_dev/background_config.yaml',
    '~/.omni_dev/background_config.yaml'
]


def auto_load_config() -> BackgroundTaskConfig:
    """
    Automatically load configuration from default locations.
    
    Priority:
    1. Environment variables
    2. Local config files
    3. System config files
    4. Default configuration
    """
    # First check for environment variable pointing to config file
    config_file = os.getenv('OMNI_DEV_BG_CONFIG')
    if config_file and Path(config_file).exists():
        return load_config(config_file)
    
    # Check if we should load from environment
    if os.getenv('OMNI_DEV_BG_USE_ENV', '').lower() == 'true':
        return load_config('env')
    
    # Check default config file locations
    for config_path in DEFAULT_CONFIG_PATHS:
        expanded_path = Path(config_path).expanduser()
        if expanded_path.exists():
            return load_config(str(expanded_path))
    
    # Fall back to default configuration
    logger.info("Using default background task configuration")
    return load_config(None)