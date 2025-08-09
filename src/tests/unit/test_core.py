"""
Unit tests for AI Vision Core module
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from components.ai_vision.core import (
    AIVisionAgent, 
    VisionTask, 
    VisionResult, 
    TaskStatus
)


class MockVisionTask(VisionTask):
    """Mock implementation of VisionTask for testing"""
    
    def __init__(self, task_id: str, task_type: str, should_fail: bool = False, **kwargs):
        super().__init__(task_id, task_type, **kwargs)
        self.should_fail = should_fail
    
    async def execute(self, input_data):
        """Mock execution that can succeed or fail"""
        if self.should_fail:
            raise RuntimeError("Mock task failure")
        
        return VisionResult(
            task_id=self.task_id,
            status=TaskStatus.COMPLETED,
            data={"processed": input_data, "mock": True},
            confidence=0.95,
            metadata={"test": True}
        )
    
    def validate_input(self, input_data):
        """Mock validation that accepts non-None input"""
        if input_data is None:
            return False
        if isinstance(input_data, dict) and input_data.get("invalid"):
            return False
        return True


@pytest.mark.unit
@pytest.mark.vision
class TestTaskStatus:
    """Test TaskStatus enum"""
    
    def test_task_status_values(self):
        """Test that TaskStatus has expected values"""
        assert TaskStatus.PENDING.value == "pending"
        assert TaskStatus.IN_PROGRESS.value == "in_progress"
        assert TaskStatus.COMPLETED.value == "completed"
        assert TaskStatus.FAILED.value == "failed"
        assert TaskStatus.CANCELLED.value == "cancelled"


@pytest.mark.unit
@pytest.mark.vision
class TestVisionResult:
    """Test VisionResult dataclass"""
    
    def test_vision_result_creation(self):
        """Test VisionResult creation with required fields"""
        result = VisionResult(
            task_id="test_001",
            status=TaskStatus.COMPLETED,
            data={"result": "success"},
            confidence=0.95
        )
        
        assert result.task_id == "test_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.data == {"result": "success"}
        assert result.confidence == 0.95
        assert result.metadata is None
        assert result.error_message is None
    
    def test_vision_result_with_optional_fields(self):
        """Test VisionResult with optional metadata and error message"""
        result = VisionResult(
            task_id="test_002",
            status=TaskStatus.FAILED,
            data=None,
            confidence=0.0,
            metadata={"attempt": 1},
            error_message="Test error"
        )
        
        assert result.metadata == {"attempt": 1}
        assert result.error_message == "Test error"


@pytest.mark.unit
@pytest.mark.vision
class TestVisionTask:
    """Test VisionTask abstract base class"""
    
    def test_task_initialization(self):
        """Test VisionTask initialization"""
        task = MockVisionTask("task_001", "test_task", extra_param="test_value")
        
        assert task.task_id == "task_001"
        assert task.task_type == "test_task"
        assert task.status == TaskStatus.PENDING
        assert task.result is None
        assert task.metadata["extra_param"] == "test_value"
    
    def test_task_status_methods(self):
        """Test task status getter methods"""
        task = MockVisionTask("task_001", "test_task")
        
        # Initial status
        assert task.get_status() == TaskStatus.PENDING
        
        # Change status
        task.status = TaskStatus.IN_PROGRESS
        assert task.get_status() == TaskStatus.IN_PROGRESS
    
    def test_task_result_methods(self):
        """Test task result getter methods"""
        task = MockVisionTask("task_001", "test_task")
        
        # Initially no result
        assert task.get_result() is None
        
        # Set result
        result = VisionResult("task_001", TaskStatus.COMPLETED, {"data": "test"}, 0.9)
        task.result = result
        assert task.get_result() == result
    
    def test_input_validation(self):
        """Test input validation"""
        task = MockVisionTask("task_001", "test_task")
        
        # Valid inputs
        assert task.validate_input("valid_input") is True
        assert task.validate_input({"valid": "data"}) is True
        
        # Invalid inputs
        assert task.validate_input(None) is False
        assert task.validate_input({"invalid": True}) is False
    
    @pytest.mark.asyncio
    async def test_task_execution_success(self):
        """Test successful task execution"""
        task = MockVisionTask("task_001", "test_task")
        input_data = {"test": "data"}
        
        result = await task.execute(input_data)
        
        assert result.task_id == "task_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.data["processed"] == input_data
        assert result.confidence == 0.95
    
    @pytest.mark.asyncio
    async def test_task_execution_failure(self):
        """Test task execution failure"""
        task = MockVisionTask("task_001", "test_task", should_fail=True)
        
        with pytest.raises(RuntimeError, match="Mock task failure"):
            await task.execute({"test": "data"})


@pytest.mark.unit
@pytest.mark.vision
class TestAIVisionAgent:
    """Test AIVisionAgent class"""
    
    def test_agent_initialization(self, vision_agent_config):
        """Test agent initialization with config"""
        agent = AIVisionAgent(
            agent_id="test_agent_001",
            name="Test Vision Agent",
            **vision_agent_config
        )
        
        assert agent.agent_id == "test_agent_001"
        assert agent.name == "Test Vision Agent"
        assert agent.max_concurrent_tasks == 3
        assert agent.task_timeout == 300
        assert agent.enable_caching is True
        assert len(agent.tasks) == 0
    
    def test_agent_default_initialization(self):
        """Test agent initialization with default config"""
        agent = AIVisionAgent(
            agent_id="default_agent",
            name="Default Agent"
        )
        
        assert agent.max_concurrent_tasks == 5  # default
        assert agent.task_timeout == 300  # default
        assert agent.enable_caching is True  # default
    
    def test_register_task_success(self):
        """Test successful task registration"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task")
        
        result = agent.register_task(task)
        
        assert result is True
        assert len(agent.tasks) == 1
        assert "task_001" in agent.tasks
        assert agent.tasks["task_001"] == task
    
    def test_register_duplicate_task(self):
        """Test registration of duplicate task IDs"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task1 = MockVisionTask("task_001", "test_task")
        task2 = MockVisionTask("task_001", "different_task")
        
        # First registration should succeed
        result1 = agent.register_task(task1)
        assert result1 is True
        
        # Second registration should fail
        result2 = agent.register_task(task2)
        assert result2 is False
        assert len(agent.tasks) == 1
        assert agent.tasks["task_001"] == task1  # Original task preserved
    
    def test_max_concurrent_tasks_limit(self):
        """Test maximum concurrent tasks limit"""
        agent = AIVisionAgent("agent_001", "Test Agent", max_concurrent_tasks=2)
        
        # Register tasks up to the limit
        task1 = MockVisionTask("task_001", "test_task")
        task2 = MockVisionTask("task_002", "test_task")
        
        assert agent.register_task(task1) is True
        assert agent.register_task(task2) is True
        assert len(agent.tasks) == 2
        
        # Try to register one more (should fail)
        task3 = MockVisionTask("task_003", "test_task")
        result = agent.register_task(task3)
        assert result is False
        assert len(agent.tasks) == 2
    
    def test_unregister_task_success(self):
        """Test successful task unregistration"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task")
        
        # Register then unregister
        agent.register_task(task)
        assert len(agent.tasks) == 1
        
        result = agent.unregister_task("task_001")
        assert result is True
        assert len(agent.tasks) == 0
        assert "task_001" not in agent.tasks
    
    def test_unregister_nonexistent_task(self):
        """Test unregistering a task that doesn't exist"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        result = agent.unregister_task("nonexistent_task")
        assert result is False
    
    def test_get_task(self):
        """Test getting a registered task"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task")
        agent.register_task(task)
        
        # Get existing task
        retrieved_task = agent.get_task("task_001")
        assert retrieved_task is not None
        assert retrieved_task == task
        
        # Get non-existent task
        non_existent = agent.get_task("non_existent")
        assert non_existent is None
    
    def test_list_tasks_empty(self):
        """Test listing tasks when no tasks are registered"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task_list = agent.list_tasks()
        assert len(task_list) == 0
        assert isinstance(task_list, list)
    
    def test_list_tasks_with_tasks(self):
        """Test listing tasks with registered tasks"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        # Register multiple tasks
        task1 = MockVisionTask("task_001", "classification", priority="high")
        task2 = MockVisionTask("task_002", "detection", priority="low")
        
        agent.register_task(task1)
        agent.register_task(task2)
        
        task_list = agent.list_tasks()
        assert len(task_list) == 2
        
        # Check task info structure
        task_info = task_list[0]
        assert "task_id" in task_info
        assert "task_type" in task_info
        assert "status" in task_info
        assert "metadata" in task_info
        
        # Check specific values
        task_ids = [info["task_id"] for info in task_list]
        assert "task_001" in task_ids
        assert "task_002" in task_ids
    
    @pytest.mark.asyncio
    async def test_execute_task_success(self):
        """Test successful task execution"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task")
        agent.register_task(task)
        
        input_data = {"test": "data"}
        result = await agent.execute_task("task_001", input_data)
        
        assert result is not None
        assert result.task_id == "task_001"
        assert result.status == TaskStatus.COMPLETED
        assert result.data["processed"] == input_data
        assert result.confidence == 0.95
        
        # Check task state
        assert task.status == TaskStatus.COMPLETED
        assert task.result == result
    
    @pytest.mark.asyncio
    async def test_execute_nonexistent_task(self):
        """Test executing a task that doesn't exist"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        result = await agent.execute_task("nonexistent", {"data": "test"})
        assert result is None
    
    @pytest.mark.asyncio
    async def test_execute_task_validation_failure(self):
        """Test executing task with invalid input"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task")
        agent.register_task(task)
        
        # Invalid input should fail validation
        result = await agent.execute_task("task_001", None)
        
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert result.error_message == "Input validation failed"
        assert result.confidence == 0.0
        assert task.status == TaskStatus.FAILED
    
    @pytest.mark.asyncio
    async def test_execute_task_execution_failure(self):
        """Test task execution failure"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        task = MockVisionTask("task_001", "test_task", should_fail=True)
        agent.register_task(task)
        
        result = await agent.execute_task("task_001", {"valid": "data"})
        
        assert result is not None
        assert result.status == TaskStatus.FAILED
        assert "Mock task failure" in result.error_message
        assert result.confidence == 0.0
        assert task.status == TaskStatus.FAILED
    
    def test_get_agent_info(self):
        """Test getting agent information"""
        config = {"custom_param": "value", "max_concurrent_tasks": 10}
        agent = AIVisionAgent("agent_001", "Test Agent", **config)
        
        # Add some tasks
        task1 = MockVisionTask("task_001", "test_task")
        task2 = MockVisionTask("task_002", "test_task")
        agent.register_task(task1)
        agent.register_task(task2)
        
        info = agent.get_agent_info()
        
        assert "agent_id" in info
        assert "name" in info
        assert "active_tasks" in info
        assert "max_concurrent_tasks" in info
        assert "config" in info
        
        assert info["agent_id"] == "agent_001"
        assert info["name"] == "Test Agent"
        assert info["active_tasks"] == 2
        assert info["max_concurrent_tasks"] == 10
        assert info["config"]["custom_param"] == "value"
    
    def test_agent_error_handling_in_register(self):
        """Test error handling in task registration"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        # Test with None task (should handle gracefully)
        with patch.object(agent, 'logger') as mock_logger:
            result = agent.register_task(None)
            assert result is False
            mock_logger.error.assert_called()
    
    def test_agent_error_handling_in_unregister(self):
        """Test error handling in task unregistration"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        # Test with exception during unregistration
        with patch.object(agent.tasks, 'pop', side_effect=Exception("Test error")):
            with patch.object(agent, 'logger') as mock_logger:
                result = agent.unregister_task("task_001")
                assert result is False
                mock_logger.error.assert_called()


@pytest.mark.unit
@pytest.mark.vision
class TestAIVisionAgentConcurrency:
    """Test concurrent task execution in AIVisionAgent"""
    
    @pytest.mark.asyncio
    async def test_concurrent_task_execution(self):
        """Test executing multiple tasks concurrently"""
        agent = AIVisionAgent("agent_001", "Test Agent", max_concurrent_tasks=5)
        
        # Register multiple tasks
        tasks = []
        for i in range(3):
            task = MockVisionTask(f"task_{i:03d}", "test_task")
            agent.register_task(task)
            tasks.append(task)
        
        # Execute tasks concurrently
        inputs = [{"data": f"test_{i}"} for i in range(3)]
        coroutines = [
            agent.execute_task(f"task_{i:03d}", inputs[i]) 
            for i in range(3)
        ]
        
        results = await asyncio.gather(*coroutines)
        
        # All tasks should complete successfully
        assert len(results) == 3
        for i, result in enumerate(results):
            assert result is not None
            assert result.status == TaskStatus.COMPLETED
            assert result.data["processed"] == inputs[i]
    
    @pytest.mark.asyncio
    async def test_mixed_success_failure_execution(self):
        """Test concurrent execution with mixed success and failure"""
        agent = AIVisionAgent("agent_001", "Test Agent")
        
        # Register tasks with different outcomes
        success_task = MockVisionTask("success", "test_task", should_fail=False)
        failure_task = MockVisionTask("failure", "test_task", should_fail=True)
        
        agent.register_task(success_task)
        agent.register_task(failure_task)
        
        # Execute both concurrently
        success_coro = agent.execute_task("success", {"data": "good"})
        failure_coro = agent.execute_task("failure", {"data": "bad"})
        
        success_result, failure_result = await asyncio.gather(
            success_coro, failure_coro, return_exceptions=False
        )
        
        # Check results
        assert success_result.status == TaskStatus.COMPLETED
        assert failure_result.status == TaskStatus.FAILED
        assert "Mock task failure" in failure_result.error_message
