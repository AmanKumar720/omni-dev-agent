"""
Free Tier Task Handler for Omni-Dev Agent
Main orchestrator for researching, analyzing, and integrating free tier services
"""

import json
import yaml
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import logging
from pathlib import Path

from .free_tier_analyzer import FreeTierAnalyzer, FreeTierService
from .service_researcher import ServiceResearcher, ServiceInfo, IntegrationPlan
from .aws_manager import AWSManager
# from ..aws_monitor.monitor import AWSFreeTierMonitor  # Commented out for demo

@dataclass
class TaskResult:
    """Result of a free tier task"""
    service_name: str
    provider: str
    free_tier_available: bool
    free_tier_details: str
    integration_plan: Optional[IntegrationPlan]
    usage_analysis: Optional[Dict[str, Any]]
    risk_assessment: Optional[Dict[str, Any]]
    recommendations: List[str]
    estimated_monthly_cost: float
    status: str  # success, warning, error

class FreeTierTaskHandler:
    """Main handler for free tier research and integration tasks"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.logger = self._setup_logging()
        
        # Initialize components
        self.free_tier_analyzer = FreeTierAnalyzer()
        self.service_researcher = ServiceResearcher()
        self.aws_manager = AWSManager()
        # self.aws_monitor = AWSFreeTierMonitor()  # Commented out for demo
        
        # Load project configuration
        self.project_config = self._load_project_config()
    
    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the task handler"""
        logger = logging.getLogger("FreeTierTaskHandler")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler("free_tier_tasks.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger
    
    def _load_project_config(self) -> Dict[str, Any]:
        """Load project configuration"""
        config_file = self.project_root / "free_tier_project_config.yaml"
        
        if config_file.exists():
            with open(config_file, 'r') as f:
                return yaml.safe_load(f)
        
        # Create default configuration
        default_config = {
            "project_name": "omni-dev-agent",
            "budget_limit": 0.0,  # Zero budget - free tier only
            "preferred_providers": ["aws", "supabase", "vercel", "netlify"],
            "monitoring_enabled": True,
            "alert_threshold": 80.0,  # Alert when 80% of free tier is used
            "auto_optimization": True
        }
        
        with open(config_file, 'w') as f:
            yaml.dump(default_config, f, default_flow_style=False)
        
        return default_config
    
    def research_and_analyze_service(self, service_name: str, project_requirements: Dict[str, Any] = None) -> TaskResult:
        """Main method to research and analyze a service for free tier usage"""
        self.logger.info(f"Starting research and analysis for service: {service_name}")
        
        if project_requirements is None:
            project_requirements = self._get_default_project_requirements()
        
        try:
            # Step 1: Research the service
            service_info = self.service_researcher.research_service(service_name)
            if not service_info:
                return TaskResult(
                    service_name=service_name,
                    provider="unknown",
                    free_tier_available=False,
                    free_tier_details="Service not found or not supported",
                    integration_plan=None,
                    usage_analysis=None,
                    risk_assessment=None,
                    recommendations=["Service not found. Please check the service name."],
                    estimated_monthly_cost=0.0,
                    status="error"
                )
            
            # Step 2: Analyze free tier availability
            free_tier_service = self.free_tier_analyzer.research_service_free_tier(service_name, service_info.provider.lower())
            
            # Step 3: Generate integration plan
            integration_plan = self.service_researcher.analyze_service_for_project(service_name, project_requirements)
            
            # Step 4: Analyze current usage (if applicable)
            usage_analysis = self._analyze_current_usage(service_name, service_info)
            
            # Step 5: Assess risks
            risk_assessment = self._assess_risks(service_name, free_tier_service, usage_analysis)
            
            # Step 6: Generate recommendations
            recommendations = self._generate_recommendations(service_info, free_tier_service, integration_plan, risk_assessment)
            
            # Step 7: Calculate estimated costs
            estimated_cost = self._calculate_estimated_cost(free_tier_service, usage_analysis)
            
            return TaskResult(
                service_name=service_name,
                provider=service_info.provider,
                free_tier_available=service_info.free_tier_available,
                free_tier_details=service_info.free_tier_details or "No free tier information available",
                integration_plan=integration_plan,
                usage_analysis=usage_analysis,
                risk_assessment=risk_assessment,
                recommendations=recommendations,
                estimated_monthly_cost=estimated_cost,
                status="success" if service_info.free_tier_available else "warning"
            )
            
        except Exception as e:
            self.logger.error(f"Error analyzing service {service_name}: {e}")
            return TaskResult(
                service_name=service_name,
                provider="unknown",
                free_tier_available=False,
                free_tier_details=f"Error during analysis: {str(e)}",
                integration_plan=None,
                usage_analysis=None,
                risk_assessment=None,
                recommendations=[f"Error occurred: {str(e)}"],
                estimated_monthly_cost=0.0,
                status="error"
            )
    
    def _get_default_project_requirements(self) -> Dict[str, Any]:
        """Get default project requirements"""
        return {
            "project_type": "web_development",
            "technologies": ["python", "javascript", "html", "css"],
            "features": ["database", "authentication", "api", "hosting"],
            "budget_constraint": "free_tier_only",
            "complexity": "medium"
        }
    
    def _analyze_current_usage(self, service_name: str, service_info: ServiceInfo) -> Optional[Dict[str, Any]]:
        """Analyze current usage of the service"""
        try:
            if "aws" in service_info.name.lower():
                # Use AWS monitor for AWS services
                if service_name.lower() == "ec2":
                    # usage_data = self.aws_monitor.get_ec2_free_tier_usage()  # Commented out for demo
                    usage_data = {"demo": "AWS EC2 usage data would be retrieved here"}
                    return {
                        "service": "ec2",
                        "usage_data": usage_data,
                        "last_updated": datetime.now().isoformat()
                    }
                # Add other AWS services as needed
            
            # For other services, return basic structure
            return {
                "service": service_name,
                "usage_data": {},
                "last_updated": datetime.now().isoformat(),
                "note": "Usage monitoring not implemented for this service"
            }
            
        except Exception as e:
            self.logger.warning(f"Could not analyze usage for {service_name}: {e}")
            return None
    
    def _assess_risks(self, service_name: str, free_tier_service: Optional[FreeTierService], usage_analysis: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Assess risks of using the service"""
        if not free_tier_service:
            return {
                "risk_level": "UNKNOWN",
                "risks": ["Free tier information not available"],
                "mitigation": ["Research service pricing before use"]
            }
        
        risks = []
        mitigation = []
        
        # Check if service has usage limits
        if free_tier_service.limit_value > 0:
            risks.append(f"Usage limited to {free_tier_service.limit_value} {free_tier_service.limit_unit}")
            mitigation.append("Monitor usage closely and implement alerts")
        
        # Check if service has time limits
        if free_tier_service.time_period == "monthly":
            risks.append("Free tier resets monthly")
            mitigation.append("Plan usage to stay within monthly limits")
        
        # Check cost implications
        if free_tier_service.cost_per_unit:
            risks.append(f"Costs ${free_tier_service.cost_per_unit} per {free_tier_service.limit_unit} after limit")
            mitigation.append("Set up billing alerts and usage monitoring")
        
        return {
            "risk_level": "LOW" if len(risks) <= 1 else "MEDIUM",
            "risks": risks,
            "mitigation": mitigation
        }
    
    def _generate_recommendations(self, service_info: ServiceInfo, free_tier_service: Optional[FreeTierService], 
                                integration_plan: Optional[IntegrationPlan], risk_assessment: Optional[Dict[str, Any]]) -> List[str]:
        """Generate recommendations for using the service"""
        recommendations = []
        
        # Basic recommendations
        if service_info.free_tier_available:
            recommendations.append(f"✅ {service_info.name} offers free tier: {service_info.free_tier_details}")
        else:
            recommendations.append(f"⚠️ {service_info.name} does not offer free tier")
        
        # Integration recommendations
        if integration_plan:
            recommendations.append(f"📋 Integration effort: {integration_plan.estimated_effort}")
            recommendations.append(f"🔐 Required credentials: {', '.join(integration_plan.required_credentials)}")
        
        # Risk-based recommendations
        if risk_assessment and risk_assessment.get("mitigation"):
            recommendations.extend([f"🛡️ {mitigation}" for mitigation in risk_assessment["mitigation"]])
        
        # Cost recommendations
        if free_tier_service and free_tier_service.cost_per_unit:
            recommendations.append(f"💰 Monitor costs: ${free_tier_service.cost_per_unit} per {free_tier_service.limit_unit} after limit")
        
        # Documentation recommendations
        if service_info.documentation_url:
            recommendations.append(f"📚 Documentation available at: {service_info.documentation_url}")
        
        return recommendations
    
    def _calculate_estimated_cost(self, free_tier_service: Optional[FreeTierService], usage_analysis: Optional[Dict[str, Any]]) -> float:
        """Calculate estimated monthly cost"""
        if not free_tier_service or not free_tier_service.cost_per_unit:
            return 0.0
        
        # This is a simplified calculation
        # In a real implementation, you'd analyze actual usage patterns
        estimated_usage = free_tier_service.limit_value * 0.8  # Assume 80% of limit
        if estimated_usage <= free_tier_service.limit_value:
            return 0.0
        else:
            excess_usage = estimated_usage - free_tier_service.limit_value
            return excess_usage * free_tier_service.cost_per_unit
    
    def get_recommended_services_for_project(self, project_requirements: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Get recommended services for a project"""
        if project_requirements is None:
            project_requirements = self._get_default_project_requirements()
        
        recommendations = self.service_researcher.get_recommended_services(project_requirements)
        
        results = []
        for service_name, score in recommendations[:5]:  # Top 5 recommendations
            service_info = self.service_researcher.research_service(service_name)
            if service_info:
                results.append({
                    "service_name": service_name,
                    "provider": service_info.provider,
                    "suitability_score": score,
                    "free_tier_available": service_info.free_tier_available,
                    "free_tier_details": service_info.free_tier_details,
                    "website_url": service_info.website_url
                })
        
        return results
    
    def generate_integration_report(self, service_name: str) -> str:
        """Generate a detailed integration report for a service"""
        task_result = self.research_and_analyze_service(service_name)
        
        report = f"""
# Free Tier Integration Report: {service_name}

## Service Information
- **Provider**: {task_result.provider}
- **Free Tier Available**: {'✅ Yes' if task_result.free_tier_available else '❌ No'}
- **Free Tier Details**: {task_result.free_tier_details}

## Integration Plan
"""
        
        if task_result.integration_plan:
            report += f"""
- **Estimated Effort**: {task_result.integration_plan.estimated_effort}
- **Risk Level**: {task_result.integration_plan.risk_level}
- **Cost Implications**: {task_result.integration_plan.cost_implications}

### Integration Steps:
"""
            for i, step in enumerate(task_result.integration_plan.integration_steps, 1):
                report += f"{i}. {step}\n"
            
            report += f"""
### Required Credentials:
- {chr(10).join(f'- {cred}' for cred in task_result.integration_plan.required_credentials)}

### Configuration Files:
- {chr(10).join(f'- {file}' for file in task_result.integration_plan.configuration_files)}
"""
        
        report += f"""
## Risk Assessment
"""
        
        if task_result.risk_assessment:
            report += f"""
- **Risk Level**: {task_result.risk_assessment.get('risk_level', 'Unknown')}
- **Risks**: {chr(10).join(f'  - {risk}' for risk in task_result.risk_assessment.get('risks', []))}
- **Mitigation**: {chr(10).join(f'  - {mitigation}' for mitigation in task_result.risk_assessment.get('mitigation', []))}
"""
        
        report += f"""
## Recommendations
{chr(10).join(f'- {rec}' for rec in task_result.recommendations)}

## Cost Analysis
- **Estimated Monthly Cost**: ${task_result.estimated_monthly_cost:.2f}
- **Status**: {task_result.status.upper()}

---
*Report generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*
"""
        
        return report
    
    def save_task_result(self, task_result: TaskResult, filename: str = None) -> str:
        """Save task result to file"""
        if filename is None:
            filename = f"free_tier_analysis_{task_result.service_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        output_file = self.project_root / "reports" / filename
        output_file.parent.mkdir(exist_ok=True)
        
        # Convert dataclass to dict for JSON serialization
        result_dict = asdict(task_result)
        
        # Handle datetime objects
        if task_result.usage_analysis and "last_updated" in task_result.usage_analysis:
            result_dict["usage_analysis"]["last_updated"] = task_result.usage_analysis["last_updated"]
        
        with open(output_file, 'w') as f:
            json.dump(result_dict, f, indent=2)
        
        self.logger.info(f"Task result saved to {output_file}")
        return str(output_file) 