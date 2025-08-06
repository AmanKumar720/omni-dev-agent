"""
Service Researcher Component for Omni-Dev Agent
Researches cloud services, analyzes their offerings, and integrates them into projects
"""

import requests
import json
import yaml
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
import logging
from pathlib import Path
from bs4 import BeautifulSoup
import re

@dataclass
class ServiceInfo:
    """Information about a cloud service"""
    name: str
    provider: str
    description: str
    website_url: str
    pricing_url: Optional[str] = None
    documentation_url: Optional[str] = None
    api_documentation_url: Optional[str] = None
    free_tier_available: bool = False
    free_tier_details: Optional[str] = None

@dataclass
class IntegrationPlan:
    """Plan for integrating a service into a project"""
    service_name: str
    integration_steps: List[str]
    required_credentials: List[str]
    configuration_files: List[str]
    estimated_effort: str
    risk_level: str
    cost_implications: str

class ServiceResearcher:
    """Researches and analyzes cloud services for integration"""
    
    def __init__(self, cache_dir: str = "service_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        self.logger = self._setup_logging()
        self.known_services = self._load_known_services()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the researcher"""
        logger = logging.getLogger("ServiceResearcher")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("service_research.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_known_services(self) -> Dict[str, ServiceInfo]:
        """Load known services from cache or create default list"""
        cache_file = self.cache_dir / "known_services.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                data = json.load(f)
                return {name: ServiceInfo(**info) for name, info in data.items()}
        
        # Default known services
        default_services = {
            "aws": ServiceInfo(
                name="Amazon Web Services",
                provider="Amazon",
                description="Comprehensive cloud computing platform",
                website_url="https://aws.amazon.com",
                pricing_url="https://aws.amazon.com/pricing/",
                documentation_url="https://docs.aws.amazon.com/",
                free_tier_available=True,
                free_tier_details="12 months free tier with 40+ services"
            ),
            "supabase": ServiceInfo(
                name="Supabase",
                provider="Supabase",
                description="Open source Firebase alternative",
                website_url="https://supabase.com",
                pricing_url="https://supabase.com/pricing",
                documentation_url="https://supabase.com/docs",
                free_tier_available=True,
                free_tier_details="Free tier with 500MB database, 50K users"
            ),
            "vercel": ServiceInfo(
                name="Vercel",
                provider="Vercel",
                description="Platform for static sites and serverless functions",
                website_url="https://vercel.com",
                pricing_url="https://vercel.com/pricing",
                documentation_url="https://vercel.com/docs",
                free_tier_available=True,
                free_tier_details="Free tier with 100GB bandwidth, serverless functions"
            ),
            "netlify": ServiceInfo(
                name="Netlify",
                provider="Netlify",
                description="Web developer platform for static sites",
                website_url="https://netlify.com",
                pricing_url="https://netlify.com/pricing",
                documentation_url="https://docs.netlify.com",
                free_tier_available=True,
                free_tier_details="Free tier with 100GB bandwidth, form submissions"
            )
        }
        
        self._save_known_services(default_services)
        return default_services
    
    def _save_known_services(self, services: Dict[str, ServiceInfo]):
        """Save known services to cache"""
        cache_file = self.cache_dir / "known_services.json"
        data = {name: {
            "name": service.name,
            "provider": service.provider,
            "description": service.description,
            "website_url": service.website_url,
            "pricing_url": service.pricing_url,
            "documentation_url": service.documentation_url,
            "api_documentation_url": service.api_documentation_url,
            "free_tier_available": service.free_tier_available,
            "free_tier_details": service.free_tier_details
        } for name, service in services.items()}
        
        with open(cache_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def research_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Research a specific service by name"""
        self.logger.info(f"Researching service: {service_name}")
        
        # Check if we already know about this service
        if service_name.lower() in self.known_services:
            return self.known_services[service_name.lower()]
        
        # Try to find service by searching known services
        for name, service in self.known_services.items():
            if service_name.lower() in name.lower() or service_name.lower() in service.name.lower():
                return service
        
        # If not found, attempt web research (placeholder for future implementation)
        return self._web_research_service(service_name)
    
    def _web_research_service(self, service_name: str) -> Optional[ServiceInfo]:
        """Research service through web scraping (placeholder implementation)"""
        self.logger.info(f"Attempting web research for: {service_name}")
        
        # This would implement web scraping to find service information
        # For now, return None to indicate service not found
        return None
    
    def analyze_service_for_project(self, service_name: str, project_requirements: Dict[str, Any]) -> Optional[IntegrationPlan]:
        """Analyze if a service is suitable for a project"""
        service_info = self.research_service(service_name)
        if not service_info:
            return None
        
        # Analyze service suitability based on project requirements
        suitability_score = self._calculate_suitability_score(service_info, project_requirements)
        
        if suitability_score < 0.5:
            return None
        
        # Generate integration plan
        integration_steps = self._generate_integration_steps(service_info, project_requirements)
        required_credentials = self._identify_required_credentials(service_info)
        configuration_files = self._identify_configuration_files(service_info)
        
        return IntegrationPlan(
            service_name=service_name,
            integration_steps=integration_steps,
            required_credentials=required_credentials,
            configuration_files=configuration_files,
            estimated_effort=self._estimate_integration_effort(service_info),
            risk_level=self._assess_integration_risk(service_info),
            cost_implications=self._analyze_cost_implications(service_info)
        )
    
    def _calculate_suitability_score(self, service_info: ServiceInfo, requirements: Dict[str, Any]) -> float:
        """Calculate how suitable a service is for the project requirements"""
        score = 0.0
        
        # Check if service has free tier (bonus points)
        if service_info.free_tier_available:
            score += 0.3
        
        # Check if service matches project type
        project_type = requirements.get("project_type", "").lower()
        service_name = service_info.name.lower()
        
        if "web" in project_type and any(word in service_name for word in ["hosting", "vercel", "netlify"]):
            score += 0.4
        elif "database" in project_type and any(word in service_name for word in ["database", "supabase", "firebase"]):
            score += 0.4
        elif "api" in project_type and any(word in service_name for word in ["api", "lambda", "functions"]):
            score += 0.4
        
        # Check if service has good documentation
        if service_info.documentation_url:
            score += 0.2
        
        # Check if service has API documentation
        if service_info.api_documentation_url:
            score += 0.1
        
        return min(score, 1.0)
    
    def _generate_integration_steps(self, service_info: ServiceInfo, requirements: Dict[str, Any]) -> List[str]:
        """Generate steps for integrating the service"""
        steps = []
        
        # Generic integration steps
        steps.append(f"Research {service_info.name} documentation")
        steps.append(f"Create account on {service_info.website_url}")
        
        if service_info.free_tier_available:
            steps.append(f"Set up {service_info.name} free tier account")
        
        # Service-specific steps
        if "supabase" in service_info.name.lower():
            steps.extend([
                "Create new Supabase project",
                "Set up database schema",
                "Configure authentication",
                "Install Supabase client library",
                "Initialize Supabase client in project"
            ])
        elif "aws" in service_info.name.lower():
            steps.extend([
                "Create AWS account",
                "Set up IAM user with appropriate permissions",
                "Install AWS CLI",
                "Configure AWS credentials",
                "Install boto3 library"
            ])
        elif "vercel" in service_info.name.lower():
            steps.extend([
                "Connect GitHub repository to Vercel",
                "Configure build settings",
                "Set up environment variables",
                "Deploy application"
            ])
        
        return steps
    
    def _identify_required_credentials(self, service_info: ServiceInfo) -> List[str]:
        """Identify required credentials for the service"""
        credentials = []
        
        if "supabase" in service_info.name.lower():
            credentials.extend(["SUPABASE_URL", "SUPABASE_ANON_KEY"])
        elif "aws" in service_info.name.lower():
            credentials.extend(["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"])
        elif "vercel" in service_info.name.lower():
            credentials.extend(["VERCEL_TOKEN"])
        
        return credentials
    
    def _identify_configuration_files(self, service_info: ServiceInfo) -> List[str]:
        """Identify configuration files needed for the service"""
        config_files = []
        
        if "supabase" in service_info.name.lower():
            config_files.extend([".env", "supabase/config.toml"])
        elif "aws" in service_info.name.lower():
            config_files.extend([".env", "~/.aws/credentials", "~/.aws/config"])
        elif "vercel" in service_info.name.lower():
            config_files.extend(["vercel.json", ".env"])
        
        return config_files
    
    def _estimate_integration_effort(self, service_info: ServiceInfo) -> str:
        """Estimate effort required for integration"""
        if "supabase" in service_info.name.lower():
            return "2-4 hours"
        elif "aws" in service_info.name.lower():
            return "4-8 hours"
        elif "vercel" in service_info.name.lower():
            return "1-2 hours"
        else:
            return "2-6 hours"
    
    def _assess_integration_risk(self, service_info: ServiceInfo) -> str:
        """Assess risk level of integrating the service"""
        if service_info.free_tier_available:
            return "LOW"
        else:
            return "MEDIUM"
    
    def _analyze_cost_implications(self, service_info: ServiceInfo) -> str:
        """Analyze cost implications of using the service"""
        if service_info.free_tier_available:
            return f"Free tier available: {service_info.free_tier_details}"
        else:
            return "No free tier - check pricing at " + (service_info.pricing_url or service_info.website_url)
    
    def get_recommended_services(self, project_requirements: Dict[str, Any]) -> List[Tuple[str, float]]:
        """Get recommended services for a project"""
        recommendations = []
        
        for service_name, service_info in self.known_services.items():
            score = self._calculate_suitability_score(service_info, project_requirements)
            if score > 0.3:  # Only recommend services with reasonable suitability
                recommendations.append((service_name, score))
        
        # Sort by score (highest first)
        recommendations.sort(key=lambda x: x[1], reverse=True)
        return recommendations
    
    def add_custom_service(self, service_info: ServiceInfo):
        """Add a custom service to the known services"""
        self.known_services[service_info.name.lower()] = service_info
        self._save_known_services(self.known_services)
        self.logger.info(f"Added custom service: {service_info.name}") 