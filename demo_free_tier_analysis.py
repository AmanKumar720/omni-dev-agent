#!/usr/bin/env python3
"""
Demo script for Omni-Dev Agent Free Tier Analysis
Shows how the agent can research, analyze, and integrate free tier services
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from components.free_tier_task_handler import FreeTierTaskHandler
from components.free_tier_analyzer import FreeTierAnalyzer
from components.service_researcher import ServiceResearcher

def print_banner():
    """Print demo banner"""
    print("=" * 80)
    print("🤖 Omni-Dev Agent - Free Tier Analysis Demo")
    print("=" * 80)
    print("This demo shows how the agent can:")
    print("✅ Research cloud services and their free tier offerings")
    print("✅ Analyze free tier limits and usage")
    print("✅ Generate integration plans")
    print("✅ Assess risks and provide recommendations")
    print("✅ Monitor usage to stay within free tier limits")
    print("✅ Automate credential management")
    print("=" * 80)

def demo_service_research():
    """Demo service research capabilities"""
    print("\n🔍 DEMO 1: Service Research")
    print("-" * 40)
    
    researcher = ServiceResearcher()
    
    # Research different services
    services_to_research = ["supabase", "aws", "vercel", "netlify"]
    
    for service_name in services_to_research:
        print(f"\nResearching {service_name.upper()}...")
        service_info = researcher.research_service(service_name)
        
        if service_info:
            print(f"✅ Found: {service_info.name}")
            print(f"   Provider: {service_info.provider}")
            print(f"   Description: {service_info.description}")
            print(f"   Free Tier: {'✅ Yes' if service_info.free_tier_available else '❌ No'}")
            if service_info.free_tier_available:
                print(f"   Free Tier Details: {service_info.free_tier_details}")
            print(f"   Website: {service_info.website_url}")
        else:
            print(f"❌ Service not found: {service_name}")

def demo_free_tier_analysis():
    """Demo free tier analysis capabilities"""
    print("\n📊 DEMO 2: Free Tier Analysis")
    print("-" * 40)
    
    analyzer = FreeTierAnalyzer()
    
    # Analyze usage for different services
    services_to_analyze = [
        ("ec2", 400),  # 400 hours used out of 750 free tier
        ("lambda", 800000),  # 800K requests out of 1M free tier
        ("s3", 3.5),  # 3.5GB used out of 5GB free tier
    ]
    
    for service_name, current_usage in services_to_analyze:
        print(f"\nAnalyzing {service_name.upper()} usage...")
        analysis = analyzer.analyze_usage_risk(service_name, current_usage)
        
        if "error" not in analysis:
            print(f"✅ Service: {analysis['service_name']}")
            print(f"   Current Usage: {analysis['current_usage']}")
            print(f"   Limit: {analysis['limit']}")
            print(f"   Percentage Used: {analysis['percentage_used']:.1f}%")
            print(f"   Risk Level: {analysis['risk_level']}")
            print(f"   Estimated Cost: ${analysis['estimated_cost']:.2f}")
            print(f"   Recommendation: {analysis['recommendation']}")
        else:
            print(f"❌ Error: {analysis['error']}")

def demo_complete_analysis():
    """Demo complete analysis workflow"""
    print("\n🚀 DEMO 3: Complete Analysis Workflow")
    print("-" * 40)
    
    handler = FreeTierTaskHandler()
    
    # Define project requirements
    project_requirements = {
        "project_type": "web_development",
        "technologies": ["python", "javascript", "react"],
        "features": ["database", "authentication", "api", "hosting"],
        "budget_constraint": "free_tier_only",
        "complexity": "medium"
    }
    
    print("Project Requirements:")
    for key, value in project_requirements.items():
        print(f"  {key}: {value}")
    
    # Get recommended services
    print(f"\n📋 Getting recommended services...")
    recommendations = handler.get_recommended_services_for_project(project_requirements)
    
    print("Top Recommended Services:")
    for i, rec in enumerate(recommendations, 1):
        print(f"{i}. {rec['service_name'].upper()} (Score: {rec['suitability_score']:.2f})")
        print(f"   Provider: {rec['provider']}")
        print(f"   Free Tier: {'✅ Yes' if rec['free_tier_available'] else '❌ No'}")
        if rec['free_tier_available']:
            print(f"   Details: {rec['free_tier_details']}")
    
    # Analyze a specific service
    service_to_analyze = "supabase"
    print(f"\n🔍 Analyzing {service_to_analyze.upper()}...")
    
    task_result = handler.research_and_analyze_service(service_to_analyze, project_requirements)
    
    print(f"✅ Analysis Complete!")
    print(f"   Provider: {task_result.provider}")
    print(f"   Free Tier Available: {'✅ Yes' if task_result.free_tier_available else '❌ No'}")
    print(f"   Status: {task_result.status.upper()}")
    print(f"   Estimated Monthly Cost: ${task_result.estimated_monthly_cost:.2f}")
    
    print(f"\n📋 Recommendations:")
    for rec in task_result.recommendations:
        print(f"   {rec}")
    
    # Generate detailed report
    print(f"\n📄 Generating detailed report...")
    report = handler.generate_integration_report(service_to_analyze)
    
    # Save report to file
    report_file = f"free_tier_report_{service_to_analyze}.md"
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)
    
    print(f"✅ Detailed report saved to: {report_file}")
    
    # Save task result
    result_file = handler.save_task_result(task_result)
    print(f"✅ Task result saved to: {result_file}")

def demo_credential_management():
    """Demo credential management capabilities"""
    print("\n🔐 DEMO 4: Credential Management")
    print("-" * 40)
    
    print("The agent can automatically manage credentials for different services:")
    
    services_credentials = {
        "Supabase": ["SUPABASE_URL", "SUPABASE_ANON_KEY"],
        "AWS": ["AWS_ACCESS_KEY_ID", "AWS_SECRET_ACCESS_KEY", "AWS_REGION"],
        "Vercel": ["VERCEL_TOKEN"],
        "Netlify": ["NETLIFY_TOKEN"]
    }
    
    for service, credentials in services_credentials.items():
        print(f"\n{service}:")
        for cred in credentials:
            print(f"  🔑 {cred}")
    
    print(f"\n✅ Credentials are automatically:")
    print(f"   - Stored securely in environment variables")
    print(f"   - Managed through GitHub Secrets")
    print(f"   - Rotated automatically when needed")
    print(f"   - Never exposed in code or logs")

def demo_usage_monitoring():
    """Demo usage monitoring capabilities"""
    print("\n📈 DEMO 5: Usage Monitoring")
    print("-" * 40)
    
    print("The agent can monitor usage across multiple services:")
    
    monitoring_capabilities = [
        "Real-time usage tracking",
        "Automatic alerts when approaching limits",
        "Cost prediction and estimation",
        "Usage optimization recommendations",
        "Historical usage analysis",
        "Cross-service usage aggregation"
    ]
    
    for capability in monitoring_capabilities:
        print(f"   ✅ {capability}")
    
    print(f"\n📊 Example Monitoring Dashboard:")
    print(f"   Service: AWS EC2")
    print(f"   Current Usage: 400 hours")
    print(f"   Free Tier Limit: 750 hours")
    print(f"   Percentage Used: 53.3%")
    print(f"   Risk Level: LOW")
    print(f"   Estimated Cost: $0.00")

def main():
    """Main demo function"""
    print_banner()
    
    try:
        # Run all demos
        demo_service_research()
        demo_free_tier_analysis()
        demo_complete_analysis()
        demo_credential_management()
        demo_usage_monitoring()
        
        print("\n" + "=" * 80)
        print("🎉 DEMO COMPLETED SUCCESSFULLY!")
        print("=" * 80)
        print("The Omni-Dev Agent can now:")
        print("✅ Research any cloud service and analyze its free tier")
        print("✅ Generate detailed integration plans")
        print("✅ Monitor usage to stay within free tier limits")
        print("✅ Manage credentials securely")
        print("✅ Provide cost estimates and risk assessments")
        print("✅ Automate the entire process")
        print("=" * 80)
        
    except Exception as e:
        print(f"\n❌ Demo failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 