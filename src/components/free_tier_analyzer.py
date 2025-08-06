"""
Free Tier Analyzer Component for Omni-Dev Agent
Analyzes and monitors free tier programs across multiple cloud providers
"""

import requests
import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime, timedelta
import logging
from pathlib import Path

@dataclass
class FreeTierService:
    """Represents a free tier service offering"""
    provider: str
    service_name: str
    free_tier_limit: str
    limit_value: float
    limit_unit: str
    time_period: str  # monthly, yearly, etc.
    description: str
    api_endpoint: Optional[str] = None
    documentation_url: Optional[str] = None
    cost_per_unit: Optional[float] = None

@dataclass
class UsageMetrics:
    """Represents current usage metrics"""
    service_name: str
    current_usage: float
    usage_unit: str
    limit: float
    limit_unit: str
    percentage_used: float
    estimated_cost: float
    last_updated: datetime

class FreeTierAnalyzer:
    """Analyzes and monitors free tier programs across cloud providers"""
    
    def __init__(self, config_path: str = "free_tier_config.yaml"):
        self.config_path = Path(config_path)
        self.free_tier_services: Dict[str, List[FreeTierService]] = {}
        self.usage_metrics: Dict[str, UsageMetrics] = {}
        self.logger = self._setup_logging()
        self._load_free_tier_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the analyzer"""
        logger = logging.getLogger("FreeTierAnalyzer")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("free_tier_analysis.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_free_tier_config(self):
        """Load free tier configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                config = yaml.safe_load(f)
                self._parse_free_tier_config(config)
        else:
            self._create_default_config()
    
    def _create_default_config(self):
        """Create default free tier configuration"""
        default_config = {
            "providers": {
                "aws": {
                    "services": [
                        {
                            "name": "ec2",
                            "free_tier_limit": "750 hours/month",
                            "limit_value": 750,
                            "limit_unit": "hours",
                            "time_period": "monthly",
                            "description": "t2.micro instances",
                            "cost_per_unit": 0.0116  # per hour for t2.micro
                        },
                        {
                            "name": "lambda",
                            "free_tier_limit": "1M requests/month",
                            "limit_value": 1000000,
                            "limit_unit": "requests",
                            "time_period": "monthly",
                            "description": "Function invocations",
                            "cost_per_unit": 0.0000002  # per request
                        },
                        {
                            "name": "s3",
                            "free_tier_limit": "5GB storage",
                            "limit_value": 5,
                            "limit_unit": "GB",
                            "time_period": "monthly",
                            "description": "Standard storage",
                            "cost_per_unit": 0.023  # per GB
                        }
                    ]
                },
                "supabase": {
                    "services": [
                        {
                            "name": "database",
                            "free_tier_limit": "500MB database",
                            "limit_value": 0.5,
                            "limit_unit": "GB",
                            "time_period": "monthly",
                            "description": "PostgreSQL database",
                            "cost_per_unit": 25.0  # per GB
                        },
                        {
                            "name": "auth",
                            "free_tier_limit": "50,000 users",
                            "limit_value": 50000,
                            "limit_unit": "users",
                            "time_period": "monthly",
                            "description": "User authentication",
                            "cost_per_unit": 0.0001  # per user
                        }
                    ]
                }
            }
        }
        
        with open(self.config_path, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        self._parse_free_tier_config(default_config)
    
    def _parse_free_tier_config(self, config: Dict):
        """Parse free tier configuration into structured data"""
        for provider, provider_config in config.get("providers", {}).items():
            self.free_tier_services[provider] = []
            for service_config in provider_config.get("services", []):
                service = FreeTierService(
                    provider=provider,
                    service_name=service_config["name"],
                    free_tier_limit=service_config["free_tier_limit"],
                    limit_value=service_config["limit_value"],
                    limit_unit=service_config["limit_unit"],
                    time_period=service_config["time_period"],
                    description=service_config["description"],
                    cost_per_unit=service_config.get("cost_per_unit")
                )
                self.free_tier_services[provider].append(service)
    
    def research_service_free_tier(self, service_name: str, provider: str = None) -> Optional[FreeTierService]:
        """Research free tier information for a specific service"""
        self.logger.info(f"Researching free tier for service: {service_name}")
        
        # This would integrate with web scraping or API calls to research services
        # For now, return from our configuration
        if provider and provider in self.free_tier_services:
            for service in self.free_tier_services[provider]:
                if service.service_name.lower() == service_name.lower():
                    return service
        
        # Search across all providers
        for provider_services in self.free_tier_services.values():
            for service in provider_services:
                if service.service_name.lower() == service_name.lower():
                    return service
        
        return None
    
    def analyze_usage_risk(self, service_name: str, current_usage: float) -> Dict[str, Any]:
        """Analyze risk of exceeding free tier limits"""
        service = self._find_service(service_name)
        if not service:
            return {"error": "Service not found"}
        
        percentage_used = (current_usage / service.limit_value) * 100
        estimated_cost = 0
        
        if percentage_used > 100 and service.cost_per_unit:
            excess_usage = current_usage - service.limit_value
            estimated_cost = excess_usage * service.cost_per_unit
        
        risk_level = "LOW"
        if percentage_used > 90:
            risk_level = "HIGH"
        elif percentage_used > 75:
            risk_level = "MEDIUM"
        
        return {
            "service_name": service_name,
            "current_usage": current_usage,
            "limit": service.limit_value,
            "percentage_used": percentage_used,
            "risk_level": risk_level,
            "estimated_cost": estimated_cost,
            "recommendation": self._get_risk_recommendation(percentage_used, risk_level)
        }
    
    def _find_service(self, service_name: str) -> Optional[FreeTierService]:
        """Find a service by name across all providers"""
        for provider_services in self.free_tier_services.values():
            for service in provider_services:
                if service.service_name.lower() == service_name.lower():
                    return service
        return None
    
    def _get_risk_recommendation(self, percentage_used: float, risk_level: str) -> str:
        """Get recommendation based on usage percentage"""
        if risk_level == "HIGH":
            return "IMMEDIATE ACTION REQUIRED: Consider reducing usage or upgrading plan"
        elif risk_level == "MEDIUM":
            return "MONITOR CLOSELY: Usage approaching limits, consider optimization"
        else:
            return "SAFE: Usage well within free tier limits"
    
    def get_free_tier_summary(self) -> Dict[str, Any]:
        """Get summary of all free tier services"""
        summary = {
            "total_providers": len(self.free_tier_services),
            "total_services": sum(len(services) for services in self.free_tier_services.values()),
            "providers": {}
        }
        
        for provider, services in self.free_tier_services.items():
            summary["providers"][provider] = {
                "service_count": len(services),
                "services": [service.service_name for service in services]
            }
        
        return summary
    
    def add_custom_service(self, service: FreeTierService):
        """Add a custom service to the free tier configuration"""
        if service.provider not in self.free_tier_services:
            self.free_tier_services[service.provider] = []
        
        self.free_tier_services[service.provider].append(service)
        self._save_config()
    
    def _save_config(self):
        """Save current configuration to file"""
        config = {"providers": {}}
        
        for provider, services in self.free_tier_services.items():
            config["providers"][provider] = {"services": []}
            for service in services:
                service_dict = {
                    "name": service.service_name,
                    "free_tier_limit": service.free_tier_limit,
                    "limit_value": service.limit_value,
                    "limit_unit": service.limit_unit,
                    "time_period": service.time_period,
                    "description": service.description
                }
                if service.cost_per_unit:
                    service_dict["cost_per_unit"] = service.cost_per_unit
                
                config["providers"][provider]["services"].append(service_dict)
        
        with open(self.config_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False) 