"""
Configuration Loader for LLM Overload Handler
Loads and manages configuration from YAML files and environment variables.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union, List
from pathlib import Path
from dataclasses import asdict

try:
    from .llm_overload_handler import OverloadConfig, APILimits
except ImportError:
    from llm_overload_handler import OverloadConfig, APILimits


class LLMConfigLoader:
    """Loads and manages LLM overload handler configuration"""
    
    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self.config_path = self._resolve_config_path(config_path)
        self.config_data = {}
        self.logger = logging.getLogger(__name__ + ".LLMConfigLoader")
        
        self.load_configuration()
        
    def _resolve_config_path(self, config_path: Optional[Union[str, Path]]) -> Path:
        """Resolve the configuration file path"""
        if config_path:
            return Path(config_path)
            
        # Try different locations
        possible_paths = [
            Path("src/config/llm_overload_config.yaml"),
            Path("config/llm_overload_config.yaml"),
            Path("llm_overload_config.yaml"),
            Path.cwd() / "src/config/llm_overload_config.yaml"
        ]
        
        for path in possible_paths:
            if path.exists():
                return path
                
        # If no config file found, use defaults
        self.logger.warning("No configuration file found, using defaults")
        return None
        
    def load_configuration(self):
        """Load configuration from file and environment"""
        if self.config_path and self.config_path.exists():
            try:
                with open(self.config_path, 'r') as f:
                    self.config_data = yaml.safe_load(f) or {}
                self.logger.info(f"Loaded configuration from {self.config_path}")
            except Exception as e:
                self.logger.error(f"Failed to load configuration file: {e}")
                self.config_data = {}
        else:
            self.logger.info("Using default configuration")
            self.config_data = self._get_default_config()
            
        # Override with environment variables
        self._apply_env_overrides()
        
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration if no file is found"""
        return {
            'default': {
                'context_window_size': 8192,
                'max_context_utilization': 0.9,
                'token_warning_threshold': 0.8,
                'token_critical_threshold': 0.95,
                'retry_attempts': 3,
                'base_retry_delay': 1.0,
                'max_retry_delay': 60.0,
                'circuit_breaker_threshold': 5,
                'circuit_breaker_timeout': 300,
                'enable_token_chunking': True,
                'chunk_overlap': 100
            },
            'api_limits': {
                'gpt4': {
                    'requests_per_minute': 500,
                    'tokens_per_minute': 10000,
                    'requests_per_hour': 10000,
                    'tokens_per_hour': 1000000,
                    'max_tokens_per_request': 8192,
                    'max_concurrent_requests': 10,
                    'daily_quota': 1000000
                }
            }
        }
        
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        env_mappings = {
            'LLM_CONTEXT_WINDOW_SIZE': ('default', 'context_window_size', int),
            'LLM_MAX_CONTEXT_UTILIZATION': ('default', 'max_context_utilization', float),
            'LLM_TOKEN_WARNING_THRESHOLD': ('default', 'token_warning_threshold', float),
            'LLM_TOKEN_CRITICAL_THRESHOLD': ('default', 'token_critical_threshold', float),
            'LLM_RETRY_ATTEMPTS': ('default', 'retry_attempts', int),
            'LLM_BASE_RETRY_DELAY': ('default', 'base_retry_delay', float),
            'LLM_MAX_RETRY_DELAY': ('default', 'max_retry_delay', float),
            'LLM_ENABLE_TOKEN_CHUNKING': ('default', 'enable_token_chunking', bool),
            'LLM_CHUNK_OVERLAP': ('default', 'chunk_overlap', int),
        }
        
        for env_var, (section, key, type_func) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    # Handle boolean conversion
                    if type_func == bool:
                        parsed_value = value.lower() in ('true', '1', 'yes', 'on')
                    else:
                        parsed_value = type_func(value)
                        
                    # Ensure nested structure exists
                    if section not in self.config_data:
                        self.config_data[section] = {}
                        
                    self.config_data[section][key] = parsed_value
                    self.logger.info(f"Applied environment override: {env_var}={parsed_value}")
                except (ValueError, TypeError) as e:
                    self.logger.warning(f"Invalid environment variable {env_var}={value}: {e}")
                    
    def get_overload_config(self, 
                           service_name: Optional[str] = None,
                           environment: Optional[str] = None) -> OverloadConfig:
        """
        Get OverloadConfig for a specific service and environment
        
        Args:
            service_name: Name of the service (e.g., 'vision_analysis', 'object_detection')
            environment: Environment name (e.g., 'development', 'production', 'testing')
            
        Returns:
            OverloadConfig instance with merged settings
        """
        # Start with default config
        config_dict = self.config_data.get('default', {}).copy()
        
        # Apply service-specific overrides
        if service_name and 'services' in self.config_data:
            service_config = self.config_data['services'].get(service_name, {})
            config_dict.update(service_config)
            
        # Apply environment-specific overrides  
        if environment and 'environments' in self.config_data:
            env_config = self.config_data['environments'].get(environment, {})
            config_dict.update(env_config)
            
        # Convert to OverloadConfig
        return OverloadConfig(**{
            k: v for k, v in config_dict.items() 
            if k in OverloadConfig.__dataclass_fields__
        })
        
    def get_api_limits(self, api_provider: str = 'gpt4') -> APILimits:
        """
        Get API limits for a specific provider
        
        Args:
            api_provider: API provider name (e.g., 'gpt4', 'gpt3_5', 'claude')
            
        Returns:
            APILimits instance
        """
        api_limits_config = self.config_data.get('api_limits', {})
        provider_config = api_limits_config.get(api_provider, {})
        
        # Use defaults if provider not found
        if not provider_config:
            self.logger.warning(f"API provider '{api_provider}' not found in config, using defaults")
            provider_config = api_limits_config.get('gpt4', {})
            
        return APILimits(**{
            k: v for k, v in provider_config.items()
            if k in APILimits.__dataclass_fields__
        })
        
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring and alerting configuration"""
        return self.config_data.get('monitoring', {
            'enable_detailed_logging': True,
            'log_level': 'INFO',
            'collect_metrics': True,
            'metrics_retention_hours': 24,
            'alerts': {
                'high_token_utilization': 0.8,
                'frequent_overloads': 5,
                'circuit_breaker_opened': True,
                'consecutive_failures': 3
            }
        })
        
    def get_cost_config(self) -> Dict[str, Any]:
        """Get cost management configuration"""
        return self.config_data.get('cost_management', {
            'track_costs': True,
            'daily_cost_limit': 100.0,
            'hourly_cost_limit': 10.0,
            'token_costs': {
                'gpt4_input': 0.00003,
                'gpt4_output': 0.00006,
                'gpt3_5_input': 0.0000015,
                'gpt3_5_output': 0.000002
            }
        })
        
    def get_feature_flags(self) -> Dict[str, bool]:
        """Get feature flags configuration"""
        return self.config_data.get('features', {
            'enable_token_estimation': True,
            'enable_context_optimization': True,
            'enable_request_batching': True,
            'enable_priority_queuing': False,
            'enable_load_balancing': False
        })
        
    def validate_config(self) -> List[str]:
        """
        Validate the loaded configuration
        
        Returns:
            List of validation warnings/errors
        """
        issues = []
        
        # Validate default config
        default_config = self.config_data.get('default', {})
        if not default_config:
            issues.append("Missing default configuration section")
            
        # Check required fields
        required_fields = [
            'context_window_size', 'max_context_utilization',
            'token_warning_threshold', 'token_critical_threshold'
        ]
        
        for field in required_fields:
            if field not in default_config:
                issues.append(f"Missing required field: {field}")
                
        # Validate thresholds
        if 'token_warning_threshold' in default_config and 'token_critical_threshold' in default_config:
            warning = default_config['token_warning_threshold']
            critical = default_config['token_critical_threshold']
            
            if warning >= critical:
                issues.append("token_warning_threshold should be less than token_critical_threshold")
                
        # Validate utilization values
        utilization = default_config.get('max_context_utilization', 1.0)
        if not (0.1 <= utilization <= 1.0):
            issues.append("max_context_utilization should be between 0.1 and 1.0")
            
        return issues
        
    def reload_configuration(self):
        """Reload configuration from file"""
        self.logger.info("Reloading configuration")
        self.load_configuration()
        
        # Validate after reload
        issues = self.validate_config()
        if issues:
            self.logger.warning(f"Configuration issues found: {issues}")
        else:
            self.logger.info("Configuration reloaded successfully")
            
    def get_config_summary(self) -> Dict[str, Any]:
        """Get a summary of the current configuration"""
        return {
            'config_file': str(self.config_path) if self.config_path else 'Default',
            'services_configured': list(self.config_data.get('services', {}).keys()),
            'environments_configured': list(self.config_data.get('environments', {}).keys()),
            'api_providers_configured': list(self.config_data.get('api_limits', {}).keys()),
            'validation_issues': self.validate_config()
        }


# Global configuration loader instance
_global_config_loader = None

def get_config_loader(config_path: Optional[Union[str, Path]] = None) -> LLMConfigLoader:
    """Get global configuration loader instance"""
    global _global_config_loader
    
    if _global_config_loader is None:
        _global_config_loader = LLMConfigLoader(config_path)
    elif config_path and _global_config_loader.config_path != Path(config_path):
        # Create new loader if different config path requested
        _global_config_loader = LLMConfigLoader(config_path)
        
    return _global_config_loader


def create_configured_handler(service_name: Optional[str] = None,
                             environment: Optional[str] = None,
                             api_provider: str = 'gpt4') -> 'LLMOverloadHandler':
    """
    Create a configured LLM overload handler
    
    Args:
        service_name: Service name for service-specific config
        environment: Environment for environment-specific config  
        api_provider: API provider for limits configuration
        
    Returns:
        Configured LLMOverloadHandler instance
    """
    config_loader = get_config_loader()
    
    # Get configurations
    overload_config = config_loader.get_overload_config(service_name, environment)
    api_limits = config_loader.get_api_limits(api_provider)
    
    # Import here to avoid circular imports
    from .llm_overload_handler import LLMOverloadHandler
    
    # Create handler
    handler = LLMOverloadHandler(overload_config)
    handler.api_limits = api_limits
    
    return handler
