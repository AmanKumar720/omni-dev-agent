# src/components/ai_vision/tests/test_core.py

import unittest
import asyncio
from unittest.mock import Mock, patch
from ..core import AIVisionAgent, VisionTask, VisionResult, TaskStatus
from . import AIVisionTestCase


class MockVisionTask(VisionTask):
    """Mock implementation of VisionTask for testing"""
    
    async def execute(self, input_data):
        """Mock execution that always succeeds"""
        return VisionResult(
            task_id=self.task_id,
            status=TaskStatus.COMPLETED,
            data={"processed": input_data},
            confidence=0.95,
            metadata={"mock": True}
        )
    
    def validate_input(self, input_data):
        """Mock validation that accepts any non-None input"""
        return input_data is not None


class TestAIVisionAgent(AIVisionTestCase):
    """Test cases for AIVisionAgent"""
    
    def setUp(self):
        super().setUp()
        self.agent = AIVisionAgent(
            agent_id="test_agent_001",
            name="Test Vision Agent",
            max_concurrent_tasks=3,
            task_timeout=60
        )
    
    def test_agent_initialization(self):
        """Test agent initialization"""
        self.assertEqual(self.agent.agent_id, "test_agent_001")
        self.assertEqual(self.agent.name, "Test Vision Agent")
        self.assertEqual(self.agent.max_concurrent_tasks, 3)
        self.assertEqual(len(self.agent.tasks), 0)
    
    def test_register_task(self):
        """Test task registration"""
        task = MockVisionTask("task_001", "test_task")
        
        # Test successful registration
        result = self.agent.register_task(task)
        self.assertTrue(result)
        self.assertEqual(len(self.agent.tasks), 1)
        self.assertIn("task_001", self.agent.tasks)
    
    def test_register_duplicate_task(self):
        """Test registration of duplicate task IDs"""
        task1 = MockVisionTask("task_001", "test_task")
        task2 = MockVisionTask("task_001", "test_task")
        
        # First registration should succeed
        result1 = self.agent.register_task(task1)
        self.assertTrue(result1)
        
        # Second registration should fail
        result2 = self.agent.register_task(task2)
        self.assertFalse(result2)
        self.assertEqual(len(self.agent.tasks), 1)
    
    def test_max_concurrent_tasks_limit(self):
        """Test maximum concurrent tasks limit"""
        # Register tasks up to the limit
        for i in range(3):
            task = MockVisionTask(f"task_{i:03d}", "test_task")
            result = self.agent.register_task(task)
            self.assertTrue(result)
        
        # Try to register one more (should fail)
        overflow_task = MockVisionTask("overflow_task", "test_task")
        result = self.agent.register_task(overflow_task)
        self.assertFalse(result)
        self.assertEqual(len(self.agent.tasks), 3)
    
    def test_unregister_task(self):
        """Test task unregistration"""
        task = MockVisionTask("task_001", "test_task")
        
        # Register task first
        self.agent.register_task(task)
        self.assertEqual(len(self.agent.tasks), 1)
        
        # Unregister task
        result = self.agent.unregister_task("task_001")
        self.assertTrue(result)
        self.assertEqual(len(self.agent.tasks), 0)
    
    def test_unregister_nonexistent_task(self):
        """Test unregistering a task that doesn't exist"""
        result = self.agent.unregister_task("nonexistent_task")
        self.assertFalse(result)
    
    def test_get_task(self):
        """Test getting a registered task"""
        task = MockVisionTask("task_001", "test_task")
        self.agent.register_task(task)
        
        retrieved_task = self.agent.get_task("task_001")
        self.assertIsNotNone(retrieved_task)
        self.assertEqual(retrieved_task.task_id, "task_001")
        
        # Test getting non-existent task
        non_existent = self.agent.get_task("non_existent")
        self.assertIsNone(non_existent)
    
    def test_list_tasks(self):
        """Test listing all tasks"""
        # Initially empty
        task_list = self.agent.list_tasks()
        self.assertEqual(len(task_list), 0)
        
        # Add some tasks
        for i in range(2):
            task = MockVisionTask(f"task_{i:03d}", f"test_task_{i}")
            self.agent.register_task(task)
        
        task_list = self.agent.list_tasks()
        self.assertEqual(len(task_list), 2)
        
        # Check task info structure
        task_info = task_list[0]
        self.assertIn('task_id', task_info)
        self.assertIn('task_type', task_info)
        self.assertIn('status', task_info)
        self.assertIn('metadata', task_info)
    
    async def test_execute_task_success(self):
        """Test successful task execution"""
        task = MockVisionTask("task_001", "test_task")
        self.agent.register_task(task)
        
        input_data = {"test": "data"}
        result = await self.agent.execute_task("task_001", input_data)
        
        self.assertIsNotNone(result)
        self.assertEqual(result.task_id, "task_001")
        self.assertEqual(result.status, TaskStatus.COMPLETED)
        self.assertIsNotNone(result.data)
        self.assertEqual(result.confidence, 0.95)
    
    async def test_execute_nonexistent_task(self):
        """Test executing a task that doesn't exist"""
        result = await self.agent.execute_task("nonexistent", {"data": "test"})
        self.assertIsNone(result)
    
    def test_get_agent_info(self):
        """Test getting agent information"""
        info = self.agent.get_agent_info()
        
        self.assertIn('agent_id', info)
        self.assertIn('name', info)
        self.assertIn('active_tasks', info)
        self.assertIn('max_concurrent_tasks', info)
        self.assertIn('config', info)
        
        self.assertEqual(info['agent_id'], "test_agent_001")
        self.assertEqual(info['name'], "Test Vision Agent")
        self.assertEqual(info['active_tasks'], 0)
        self.assertEqual(info['max_concurrent_tasks'], 3)


class TestVisionTask(AIVisionTestCase):
    """Test cases for VisionTask abstract class"""
    
    def test_task_initialization(self):
        """Test task initialization"""
        task = MockVisionTask("task_001", "test_task", extra_param="test")
        
        self.assertEqual(task.task_id, "task_001")
        self.assertEqual(task.task_type, "test_task")
        self.assertEqual(task.status, TaskStatus.PENDING)
        self.assertIsNone(task.result)
        self.assertIn('extra_param', task.metadata)
        self.assertEqual(task.metadata['extra_param'], "test")
    
    def test_task_status_methods(self):
        """Test task status getter methods"""
        task = MockVisionTask("task_001", "test_task")
        
        # Initial status
        self.assertEqual(task.get_status(), TaskStatus.PENDING)
        
        # Change status
        task.status = TaskStatus.IN_PROGRESS
        self.assertEqual(task.get_status(), TaskStatus.IN_PROGRESS)
    
    def test_task_result_methods(self):
        """Test task result getter methods"""
        task = MockVisionTask("task_001", "test_task")
        
        # Initially no result
        self.assertIsNone(task.get_result())
        
        # Set result
        result = VisionResult("task_001", TaskStatus.COMPLETED, {"test": "data"}, 0.9)
        task.result = result
        self.assertEqual(task.get_result(), result)


def run_async_tests():
    """Helper function to run async tests"""
    async def run_tests():
        suite = unittest.TestSuite()
        
        # Add async test methods
        test_instance = TestAIVisionAgent()
        test_instance.setUp()
        
        await test_instance.test_execute_task_success()
        await test_instance.test_execute_nonexistent_task()
        
        print("Async tests completed successfully!")
    
    asyncio.run(run_tests())


if __name__ == '__main__':
    # Run regular unit tests
    unittest.main(verbosity=2, exit=False)
    
    # Run async tests separately
    print("\n" + "="*50)
    print("Running async tests...")
    run_async_tests()
