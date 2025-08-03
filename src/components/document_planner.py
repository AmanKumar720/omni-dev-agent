"""
Document Planning Component for Omni-Dev Agent

This component is responsible for generating structured plans for development tasks.
It analyzes task descriptions and creates detailed, actionable plans.
"""

import json
from typing import Dict, List, Any
from datetime import datetime
from .document_planner_models import TaskPlan, Phase, Step


class DocumentPlanner:
    """
    Generates structured plans for development tasks.
    """
    
    def __init__(self):
        self.plan_templates = self._load_plan_templates()
    
    def _load_plan_templates(self) -> Dict[str, Any]:
        """Load predefined plan templates for common task types."""
        return {
            "web_development": {
                "phases": ["Backend Development", "Frontend Development", "Integration & Testing", "Documentation"],
                "backend_steps": [
                    "Design API endpoints and data models",
                    "Set up database schema and migrations",
                    "Implement API routes and business logic",
                    "Add authentication and authorization",
                    "Write comprehensive unit tests for API",
                    "Set up API documentation (Swagger/OpenAPI)"
                ],
                "frontend_steps": [
                    "Create wireframes and UI mockups",
                    "Set up frontend project structure",
                    "Implement core UI components",
                    "Add form validation and error handling",
                    "Implement responsive design",
                    "Write frontend unit and integration tests"
                ],
                "integration_steps": [
                    "Connect frontend to backend APIs",
                    "Implement error handling and loading states",
                    "End-to-end testing with real data",
                    "Performance testing and optimization",
                    "Cross-browser compatibility testing"
                ],
                "documentation_steps": [
                    "Document API endpoints with examples",
                    "Create user guides and tutorials",
                    "Write technical specifications",
                    "Create deployment and maintenance guides"
                ],
                "effort_multipliers": {
                    "Backend Development": 1.2,
                    "Frontend Development": 1.0,
                    "Integration & Testing": 0.8,
                    "Documentation": 0.6
                }
            },
            "api_development": {
                "phases": ["API Design", "Implementation", "Testing", "Documentation"],
                "api_design_steps": [
                    "Define API specifications (OpenAPI/Swagger)",
                    "Design data models and schemas",
                    "Plan authentication and authorization",
                    "Define error handling strategies"
                ],
                "implementation_steps": [
                    "Set up project structure and dependencies",
                    "Implement core API endpoints",
                    "Add authentication middleware",
                    "Implement data validation and sanitization",
                    "Add logging and monitoring"
                ],
                "testing_steps": [
                    "Write unit tests for all endpoints",
                    "Create integration tests",
                    "Performance and load testing",
                    "Security testing and vulnerability assessment"
                ],
                "documentation_steps": [
                    "Generate interactive API documentation",
                    "Create developer guides and examples",
                    "Write deployment instructions"
                ],
                "effort_multipliers": {
                    "API Design": 0.8,
                    "Implementation": 1.3,
                    "Testing": 1.0,
                    "Documentation": 0.5
                }
            },
            "general_development": {
                "phases": ["Planning", "Implementation", "Testing", "Documentation"],
                "planning_steps": [
                    "Analyze and document requirements",
                    "Design system architecture",
                    "Create technical specifications",
                    "Define development milestones",
                    "Set up project timeline"
                ],
                "implementation_steps": [
                    "Set up development environment",
                    "Implement core functionality",
                    "Handle edge cases and error scenarios",
                    "Code review and refactoring",
                    "Performance optimization"
                ],
                "testing_steps": [
                    "Write comprehensive unit tests",
                    "Integration testing",
                    "User acceptance testing",
                    "Performance and stress testing"
                ],
                "documentation_steps": [
                    "Code documentation and comments",
                    "User documentation and guides",
                    "Deployment and maintenance guides",
                    "Create README and setup instructions"
                ],
                "effort_multipliers": {
                    "Planning": 0.7,
                    "Implementation": 1.4,
                    "Testing": 1.0,
                    "Documentation": 0.6
                }
            }
        }
    
    def generate_plan(self, task_description: str, task_type: str = "general_development") -> TaskPlan:
        """
        Generate a structured plan for the given task.
        
        Args:
            task_description: Description of the task to plan
            task_type: Type of task (web_development, general_development, etc.)
        
        Returns:
            A structured plan dictionary
        """
        # Determine task type if not specified
        if task_type == "general_development" and self._is_web_development_task(task_description):
            task_type = "web_development"
        
        template = self.plan_templates.get(task_type, self.plan_templates["general_development"])
        
        phases_list = []
        
        # Generate phases based on template
        for i, phase_name in enumerate(template["phases"], 1):
            step_descriptions = self._get_steps_for_phase(phase_name, template)
            steps = [Step(description=desc) for desc in step_descriptions]
            effort = self._estimate_effort(phase_name, step_descriptions, template, task_description)
            estimated_hours = self._convert_effort_to_hours(effort)
            dependencies = self._get_phase_dependencies(i, len(template["phases"]))
            
            phase = Phase(
                phase_number=i,
                phase_name=phase_name,
                steps=steps,
                estimated_effort=effort,
                estimated_hours=estimated_hours,
                dependencies=dependencies
            )
            phases_list.append(phase)
        
        # Add total estimation
        total_hours = sum(phase.estimated_hours for phase in phases_list)
        total_days = round(total_hours / 8, 1)  # Assuming 8-hour work days
        
        task_plan = TaskPlan(
            task_description=task_description,
            task_type=task_type,
            created_at=datetime.now().isoformat(),
            estimated_phases=len(template["phases"]),
            total_estimated_hours=total_hours,
            total_estimated_days=total_days,
            phases=phases_list
        )
        
        return task_plan
    
    def _is_web_development_task(self, description: str) -> bool:
        """Determine if a task is web development related."""
        web_keywords = [
            "web", "website", "frontend", "backend", "api", "html", "css", "javascript",
            "react", "vue", "angular", "flask", "django", "express", "form"
        ]
        description_lower = description.lower()
        return any(keyword in description_lower for keyword in web_keywords)
    
    def _get_steps_for_phase(self, phase_name: str, template: Dict[str, Any]) -> List[str]:
        """Get steps for a specific phase based on the template."""
        # Map phase names to template keys
        phase_mapping = {
            "backend development": "backend_steps",
            "frontend development": "frontend_steps", 
            "integration & testing": "integration_steps",
            "documentation": "documentation_steps",
            "planning": "planning_steps",
            "implementation": "implementation_steps",
            "testing": "testing_steps"
        }
        
        phase_key = phase_mapping.get(phase_name.lower(), None)
        if phase_key and phase_key in template:
            return template[phase_key]
        else:
            return [f"Complete {phase_name}"]
    
    def _get_phase_dependencies(self, phase_number: int, total_phases: int) -> List[int]:
        """Get dependencies for a phase (simple sequential dependency for now)."""
        if phase_number == 1:
            return []
        return [phase_number - 1]
    
    def _estimate_effort(self, phase_name: str, steps: List[str], template: Dict[str, Any], task_description: str) -> str:
        """Estimate effort for a phase based on various factors."""
        base_hours = len(steps) * 4  # Base 4 hours per step
        
        # Apply multiplier from template if available
        multipliers = template.get("effort_multipliers", {})
        multiplier = multipliers.get(phase_name, 1.0)
        base_hours *= multiplier
        
        # Adjust based on task complexity keywords
        complexity_keywords = {
            "simple": 0.7, "basic": 0.8, "standard": 1.0,
            "complex": 1.4, "advanced": 1.6, "enterprise": 1.8,
            "authentication": 1.3, "security": 1.4, "performance": 1.2,
            "real-time": 1.5, "scalable": 1.3
        }
        
        task_lower = task_description.lower()
        complexity_multiplier = 1.0
        for keyword, mult in complexity_keywords.items():
            if keyword in task_lower:
                complexity_multiplier = max(complexity_multiplier, mult)
        
        final_hours = base_hours * complexity_multiplier
        
        # Convert to effort categories
        if final_hours <= 8:
            return "Low"
        elif final_hours <= 24:
            return "Medium"
        elif final_hours <= 48:
            return "High"
        else:
            return "Very High"
    
    def _convert_effort_to_hours(self, effort_level: str) -> int:
        """Convert effort level to estimated hours."""
        effort_mapping = {
            "Very Low": 4,
            "Low": 8,
            "Medium": 16,
            "High": 32,
            "Very High": 48
        }
        return effort_mapping.get(effort_level, 16)
    
    def format_plan_as_markdown(self, plan: Dict[str, Any]) -> str:
        """Format the plan as a Markdown document."""
        total_hours = plan.get('total_estimated_hours', 0)
        total_days = plan.get('total_estimated_days', 0)
        
        markdown = f"""# Development Plan: {plan['task_description']}

**Task Type:** {plan['task_type']}  
**Created:** {plan['created_at']}  
**Estimated Phases:** {plan['estimated_phases']}  
**Total Estimated Hours:** {total_hours}  
**Total Estimated Days:** {total_days}  

---

"""
        
        for phase in plan['phases']:
            markdown += f"## Phase {phase['phase_number']}: {phase['phase_name']}\n\n"
            markdown += f"**Estimated Effort:** {phase['estimated_effort']}  \n"
            markdown += f"**Estimated Hours:** {phase.get('estimated_hours', 0)}  \n"
            
            if phase['dependencies']:
                deps = ", ".join([f"Phase {dep}" for dep in phase['dependencies']])
                markdown += f"**Dependencies:** {deps}  \n"
            
            markdown += "\n**Steps:**\n"
            for i, step in enumerate(phase['steps'], 1):
                markdown += f"{i}. {step}\n"
            markdown += "\n---\n\n"
        
        return markdown
    
    def save_plan(self, plan: Dict[str, Any], filename: str = None) -> str:
        """Save the plan to a file."""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"plan_{timestamp}.md"
        
        markdown_content = self.format_plan_as_markdown(plan)
        
        try:
            with open(filename, 'w', encoding='utf-8') as f:
                f.write(markdown_content)
            return f"Plan saved to {filename}"
        except Exception as e:
            return f"Error saving plan: {str(e)}"


# Example usage and testing
if __name__ == "__main__":
    planner = DocumentPlanner()
    
    # Test with a web development task
    task = "Develop a web feedback form feature with backend API and frontend interface"
    plan = planner.generate_plan(task)
    
    print("Generated Plan:")
    print(json.dumps(plan, indent=2))
    
    print("\n" + "="*50 + "\n")
    print("Markdown Format:")
    print(planner.format_plan_as_markdown(plan))
