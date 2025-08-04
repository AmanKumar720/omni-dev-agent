"""
Omni-Dev Agent Continuous Learning Engine
Enables the agent to learn from experiences and improve integration skills over time.
"""

import json
import pickle
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import logging

@dataclass
class LearningExample:
    """Represents a learning example from agent's experience."""
    timestamp: datetime
    context: Dict[str, Any]
    action: str
    outcome: str
    success: bool
    feedback_score: float  # 0.0 to 1.0
    metadata: Dict[str, Any]

class LearningEngine:
    """Main learning engine that processes experiences and improves decision making."""
    
    def __init__(self, storage_path: str = "learning_data"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(exist_ok=True)
        
        self.experiences: List[LearningExample] = []
        self.action_patterns: Dict[str, Dict[str, float]] = {}
        self.success_rates: Dict[str, float] = {}
        self.context_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
        self.logger = self._setup_logging()
        self._load_existing_data()

    def _setup_logging(self) -> logging.Logger:
        """Set up logging for the learning engine."""
        logger = logging.getLogger("LearningEngine")
        logger.setLevel(logging.INFO)
        
        handler = logging.FileHandler(self.storage_path / "learning.log")
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
        return logger

    def _load_existing_data(self):
        """Load existing learning data from storage."""
        experiences_file = self.storage_path / "experiences.json"
        if experiences_file.exists():
            try:
                with open(experiences_file, 'r') as f:
                    data = json.load(f)
                    self.experiences = [
                        LearningExample(**exp) for exp in data
                    ]
                self.logger.info(f"Loaded {len(self.experiences)} existing experiences")
            except Exception as e:
                self.logger.error(f"Failed to load existing experiences: {e}")

        self._analyze_experiences()

    def record_experience(self, context: Dict[str, Any], action: str, 
                         outcome: str, success: bool, feedback_score: float = 0.5,
                         metadata: Dict[str, Any] = None):
        """Record a new learning experience."""
        experience = LearningExample(
            timestamp=datetime.now(),
            context=context,
            action=action,
            outcome=outcome,
            success=success,
            feedback_score=feedback_score,
            metadata=metadata or {}
        )
        
        self.experiences.append(experience)
        self.logger.info(f"Recorded new experience: {action} -> {outcome} (success: {success})")
        
        # Update patterns
        self._update_patterns(experience)
        
        # Save to storage
        self._save_experiences()

    def _update_patterns(self, experience: LearningExample):
        """Update learned patterns based on new experience."""
        action = experience.action
        
        # Update action patterns
        if action not in self.action_patterns:
            self.action_patterns[action] = {}
        
        outcome = experience.outcome
        if outcome not in self.action_patterns[action]:
            self.action_patterns[action][outcome] = 0.0
        
        # Weight recent experiences more heavily
        weight = 1.0 if experience.success else 0.1
        self.action_patterns[action][outcome] += weight

        # Update success rates
        action_experiences = [exp for exp in self.experiences if exp.action == action]
        successful = sum(1 for exp in action_experiences if exp.success)
        self.success_rates[action] = successful / len(action_experiences)

        # Update context patterns
        context_key = self._get_context_signature(experience.context)
        if context_key not in self.context_patterns:
            self.context_patterns[context_key] = []
        self.context_patterns[context_key].append({
            'action': action,
            'success': experience.success,
            'feedback_score': experience.feedback_score
        })

    def _get_context_signature(self, context: Dict[str, Any]) -> str:
        """Generate a signature for a context to identify similar situations."""
        # Simple signature based on key context elements
        key_elements = ['component_type', 'integration_type', 'complexity_level']
        signature_parts = []
        
        for key in key_elements:
            if key in context:
                signature_parts.append(f"{key}:{context[key]}")
        
        return "|".join(signature_parts) if signature_parts else "unknown_context"

    def recommend_action(self, context: Dict[str, Any]) -> Tuple[str, float]:
        """Recommend the best action based on learned patterns."""
        context_signature = self._get_context_signature(context)
        
        # Look for similar contexts
        if context_signature in self.context_patterns:
            patterns = self.context_patterns[context_signature]
            
            # Calculate scores for each possible action
            action_scores = {}
            for pattern in patterns:
                action = pattern['action']
                if action not in action_scores:
                    action_scores[action] = []
                
                # Consider both success and feedback score
                score = (1.0 if pattern['success'] else 0.0) * pattern['feedback_score']
                action_scores[action].append(score)
            
            # Find action with highest average score
            best_action = None
            best_score = 0.0
            
            for action, scores in action_scores.items():
                avg_score = np.mean(scores)
                if avg_score > best_score:
                    best_score = avg_score
                    best_action = action
            
            if best_action:
                confidence = min(len(action_scores[best_action]) / 10.0, 1.0)  # More examples = higher confidence
                self.logger.info(f"Recommending action '{best_action}' with confidence {confidence:.2f}")
                return best_action, confidence

        # Fallback to action with highest overall success rate
        if self.success_rates:
            best_action = max(self.success_rates, key=self.success_rates.get)
            confidence = self.success_rates[best_action]
            return best_action, confidence

        # No learning data available
        return "default_action", 0.0

    def _analyze_experiences(self):
        """Analyze all experiences to update patterns."""
        for experience in self.experiences:
            self._update_patterns(experience)

    def _save_experiences(self):
        """Save experiences to persistent storage."""
        experiences_file = self.storage_path / "experiences.json"
        try:
            # Convert datetime objects to strings for JSON serialization
            experiences_data = []
            for exp in self.experiences:
                exp_dict = asdict(exp)
                exp_dict['timestamp'] = exp.timestamp.isoformat()
                experiences_data.append(exp_dict)
            
            with open(experiences_file, 'w') as f:
                json.dump(experiences_data, f, indent=2)
                
            self.logger.info(f"Saved {len(self.experiences)} experiences to storage")
        except Exception as e:
            self.logger.error(f"Failed to save experiences: {e}")

    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from the learning process."""
        if not self.experiences:
            return {"message": "No learning experiences recorded yet"}

        total_experiences = len(self.experiences)
        successful_experiences = sum(1 for exp in self.experiences if exp.success)
        overall_success_rate = successful_experiences / total_experiences

        # Most successful actions
        sorted_actions = sorted(self.success_rates.items(), key=lambda x: x[1], reverse=True)
        
        # Learning trends
        recent_experiences = self.experiences[-50:] if len(self.experiences) > 50 else self.experiences
        recent_success_rate = sum(1 for exp in recent_experiences if exp.success) / len(recent_experiences)
        
        improvement = recent_success_rate - overall_success_rate

        return {
            "total_experiences": total_experiences,
            "overall_success_rate": overall_success_rate,
            "recent_success_rate": recent_success_rate,
            "improvement_trend": improvement,
            "most_successful_actions": sorted_actions[:5],
            "context_patterns_learned": len(self.context_patterns),
            "recommendations": self._generate_learning_recommendations()
        }

    def _generate_learning_recommendations(self) -> List[str]:
        """Generate recommendations for improving learning."""
        recommendations = []
        
        if len(self.experiences) < 50:
            recommendations.append("Collect more experiences to improve learning accuracy")
        
        if self.success_rates:
            worst_action = min(self.success_rates, key=self.success_rates.get)
            if self.success_rates[worst_action] < 0.3:
                recommendations.append(f"Review and improve strategy for '{worst_action}' action")
        
        if len(self.context_patterns) < 10:
            recommendations.append("Diversify integration contexts to learn broader patterns")
        
        return recommendations

    def export_learning_model(self, filename: str = "learning_model.pkl"):
        """Export the learned model for backup or transfer."""
        model_data = {
            'action_patterns': self.action_patterns,
            'success_rates': self.success_rates,
            'context_patterns': self.context_patterns
        }
        
        model_file = self.storage_path / filename
        with open(model_file, 'wb') as f:
            pickle.dump(model_data, f)
        
        self.logger.info(f"Exported learning model to {model_file}")

    def import_learning_model(self, filename: str = "learning_model.pkl"):
        """Import a previously saved learning model."""
        model_file = self.storage_path / filename
        if model_file.exists():
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)
            
            self.action_patterns.update(model_data.get('action_patterns', {}))
            self.success_rates.update(model_data.get('success_rates', {}))
            self.context_patterns.update(model_data.get('context_patterns', {}))
            
            self.logger.info(f"Imported learning model from {model_file}")
        else:
            self.logger.warning(f"Learning model file {model_file} not found")

# Global learning engine instance
global_learning_engine = LearningEngine()
