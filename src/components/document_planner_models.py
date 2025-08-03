from pydantic import BaseModel
from typing import List, Dict, Any, Optional
from datetime import datetime

class Step(BaseModel):
    description: str
    status: str = "pending"
    output: Optional[str] = None
    error: Optional[str] = None

class Phase(BaseModel):
    phase_number: int
    phase_name: str
    steps: List[Step]
    estimated_effort: str
    estimated_hours: int
    dependencies: List[int]
    status: str = "pending"

class TaskPlan(BaseModel):
    task_description: str
    task_type: str
    created_at: str
    estimated_phases: int
    total_estimated_hours: float
    total_estimated_days: float
    phases: List[Phase]
