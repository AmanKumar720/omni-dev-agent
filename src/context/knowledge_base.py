# Omni-Dev Agent Knowledge Base

The knowledge base is responsible for storing and reasoning about important project facts, decisions, and context to support agent operations.

import json
import os
from typing import Any, Dict

class KnowledgeBase:
    def __init__(self, storage_path: str):
        self.storage_path = storage_path
        self.knowledge: Dict[str, Any] = {}
        self._load()

    def _load(self):
        """Load existing knowledge from storage if available."""
        if os.path.exists(self.storage_path):
            with open(self.storage_path, 'r') as f:
                self.knowledge = json.load(f)

    def save(self):
        """Save knowledge to persistent storage."""
        with open(self.storage_path, 'w') as f:
            json.dump(self.knowledge, f, indent=4)

    def add_knowledge(self, key: str, value: Any):
        """Add new knowledge or update existing knowledge."""
        self.knowledge[key] = value
        self.save()

    def get_knowledge(self, key: str) -> Any:
        return self.knowledge.get(key)

    def reason_about(self, query: str) -> Any:
        """Perform reasoning based on available knowledge."""
        # Example: Simplistic reasoning logic for demonstration.
        if query in self.knowledge:
            return self.knowledge[query]
        return "No knowledge available for this query."
