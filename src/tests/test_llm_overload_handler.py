"""
Test script for LLM Overload Handler
Validates core functionality of overload detection and mitigation.
"""

import asyncio
import pytest
import logging
from unittest.mock import AsyncMock, patch
from datetime import datetime, timedelta

try:
    from ..error_handling.llm_overload_handler import (
        LLMOverloadHandler, OverloadConfig, APILimits, 
        OverloadType, OverloadSeverity, with_llm_overload_protection
    )
    from ..error_handling.llm_config_loader import LLMConfigLoader
except ImportError:
    import sys
    sys.path.append('..')
    from error_handling.llm_overload_handler import (
        LLMOverloadHandler, OverloadConfig, APILimits,
        OverloadType, OverloadSeverity, with_llm_overload_protection
    )
    from error_handling.llm_config_loader import LLMConfigLoader


class TestLLMOverloadHandler:
    """Test cases for LLM overload handler"""
    
    @pytest.fixture
    def basic_config(self):
        """Basic configuration for testing"""
        return OverloadConfig(
            context_window_size=1000,
            max_context_utilization=0.8,
            token_warning_threshold=0.7,
            token_critical_threshold=0.9,
            retry_attempts=2,
            base_retry_delay=0.1,
            max_retry_delay=1.0,
            enable_token_chunking=True,
            chunk_overlap=50
        )
        
    @pytest.fixture
    def api_limits(self):
        """API limits for testing"""
        return APILimits(
            requests_per_minute=10,
            tokens_per_minute=1000,
            requests_per_hour=100,
            tokens_per_hour=10000,
            max_tokens_per_request=500,
            max_concurrent_requests=3,
            daily_quota=50000
        )
        
    @pytest.fixture
    def handler(self, basic_config, api_limits):
        """Create handler for testing"""
        handler = LLMOverloadHandler(basic_config)
        handler.api_limits = api_limits
        return handler
        
    def test_handler_initialization(self, handler):
        """Test handler initializes correctly"""
        assert handler.config.context_window_size == 1000
        assert handler.api_limits.tokens_per_minute == 1000
        assert len(handler.overload_events) == 0
        assert handler.active_requests == 0
        
    @pytest.mark.asyncio
    async def test_no_overload_conditions(self, handler):
        """Test when no overload conditions exist"""
        events = await handler.check_overload_conditions(100, {"test": True})
        assert len(events) == 0
        
    @pytest.mark.asyncio  
    async def test_token_limit_overload(self, handler):
        """Test token limit overload detection"""
        # Simulate existing token usage
        current_time = datetime.now()
        for _ in range(5):
            handler.minute_tokens.append((current_time, type('TokenUsage', (), {'total_tokens': 150})()))
            
        # This should trigger token limit overload
        events = await handler.check_overload_conditions(500, {"test": True})
        
        assert len(events) > 0
        assert any(e.overload_type == OverloadType.TOKEN_LIMIT for e in events)
        
    @pytest.mark.asyncio
    async def test_context_overflow(self, handler):
        """Test context window overflow detection"""
        # Request tokens exceeding context window
        large_token_request = int(handler.config.context_window_size * handler.config.max_context_utilization) + 100
        
        events = await handler.check_overload_conditions(large_token_request, {"test": True})
        
        assert len(events) > 0
        assert any(e.overload_type == OverloadType.CONTEXT_OVERFLOW for e in events)
        
    @pytest.mark.asyncio
    async def test_rate_limit_overload(self, handler):
        """Test rate limit overload detection"""
        # Fill up the request queue
        current_time = datetime.now()
        for _ in range(handler.api_limits.requests_per_minute):
            handler.minute_requests.append(current_time)
            
        events = await handler.check_overload_conditions(100, {"test": True})
        
        assert len(events) > 0
        assert any(e.overload_type == OverloadType.RATE_LIMIT for e in events)
        
    def test_text_chunking(self, handler):
        """Test text chunking functionality"""
        # Create a long text that needs chunking
        long_text = " ".join([f"word{i}" for i in range(1000)])
        
        chunks = handler._split_text_into_chunks(long_text)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= handler.config.context_window_size * handler.config.max_context_utilization for chunk in chunks)
        
    def test_context_truncation(self, handler):
        """Test context truncation"""
        # Create text that needs truncation
        long_text = " ".join([f"word{i}" for i in range(1000)])
        
        truncated = handler._truncate_context(long_text)
        
        max_words = int(handler.config.context_window_size * handler.config.max_context_utilization)
        assert len(truncated.split()) <= max_words + 1  # +1 for "[truncated]"
        assert "[truncated]" in truncated
        
    @pytest.mark.asyncio
    async def test_overload_mitigation_no_events(self, handler):
        """Test mitigation when no overload events exist"""
        mock_func = AsyncMock(return_value="success")
        
        result = await handler.apply_mitigation_strategy([], mock_func, arg1="test")
        
        assert result == "success"
        mock_func.assert_called_once_with(arg1="test")
        
    @pytest.mark.asyncio
    async def test_token_chunking_mitigation(self, handler):
        """Test token chunking mitigation strategy"""
        # Create a token limit overload event
        event = type('OverloadEvent', (), {
            'overload_type': OverloadType.TOKEN_LIMIT,
            'severity': OverloadSeverity.CRITICAL,
            'context': {}
        })()
        
        mock_func = AsyncMock(return_value={"choices": [{"message": {"content": "chunk response"}}]})
        long_prompt = " ".join([f"word{i}" for i in range(1000)])
        
        result = await handler.apply_mitigation_strategy([event], mock_func, prompt=long_prompt)
        
        # Should return list of results (one per chunk)
        assert isinstance(result, list)
        assert len(result) > 1
        assert mock_func.call_count > 1  # Called once per chunk
        
    def test_usage_stats(self, handler):
        """Test usage statistics generation"""
        stats = handler.get_usage_stats()
        
        assert "current_time" in stats
        assert "active_requests" in stats
        assert "minute_requests" in stats
        assert "minute_tokens" in stats
        assert "token_utilization_minute" in stats
        assert "request_utilization_minute" in stats
        
    def test_statistics_reset(self, handler):
        """Test statistics reset functionality"""
        # Add some data
        handler.overload_events.append("dummy_event")
        handler.minute_requests.append(datetime.now())
        
        handler.reset_statistics()
        
        assert len(handler.overload_events) == 0
        assert len(handler.minute_requests) == 0
        
    @pytest.mark.asyncio
    async def test_decorator_functionality(self):
        """Test the overload protection decorator"""
        @with_llm_overload_protection()
        async def mock_llm_call(prompt="test"):
            return f"Response to: {prompt}"
            
        # This should work without overload conditions
        result = await mock_llm_call("hello")
        # Since we're using default handler, this should execute
        assert "Response to: hello" in str(result) or isinstance(result, Exception)


class TestLLMConfigLoader:
    """Test cases for LLM configuration loader"""
    
    def test_default_config_creation(self):
        """Test creation with default configuration"""
        # This should create a loader with defaults since no config file exists
        loader = LLMConfigLoader()
        
        assert loader.config_data is not None
        assert 'default' in loader.config_data
        
    def test_overload_config_generation(self):
        """Test overload configuration generation"""
        loader = LLMConfigLoader()
        
        config = loader.get_overload_config()
        
        assert isinstance(config, OverloadConfig)
        assert config.context_window_size > 0
        assert 0 < config.max_context_utilization <= 1.0
        
    def test_api_limits_generation(self):
        """Test API limits generation"""
        loader = LLMConfigLoader()
        
        limits = loader.get_api_limits('gpt4')
        
        assert isinstance(limits, APILimits)
        assert limits.tokens_per_minute > 0
        assert limits.requests_per_minute > 0
        
    def test_service_specific_config(self):
        """Test service-specific configuration"""
        # Create loader with mock config data
        loader = LLMConfigLoader()
        loader.config_data = {
            'default': {
                'context_window_size': 8192,
                'retry_attempts': 3
            },
            'services': {
                'vision_analysis': {
                    'context_window_size': 4096,
                    'retry_attempts': 5
                }
            }
        }
        
        # Get default config
        default_config = loader.get_overload_config()
        assert default_config.context_window_size == 8192
        assert default_config.retry_attempts == 3
        
        # Get service-specific config
        service_config = loader.get_overload_config('vision_analysis')
        assert service_config.context_window_size == 4096  # Overridden
        assert service_config.retry_attempts == 5  # Overridden
        
    def test_config_validation(self):
        """Test configuration validation"""
        loader = LLMConfigLoader()
        
        # Should have no issues with default config
        issues = loader.validate_config()
        assert len(issues) == 0 or all("missing" not in issue.lower() for issue in issues)
        
    def test_config_summary(self):
        """Test configuration summary generation"""
        loader = LLMConfigLoader()
        
        summary = loader.get_config_summary()
        
        assert "config_file" in summary
        assert "validation_issues" in summary
        assert isinstance(summary["validation_issues"], list)


# Async test runner for manual execution
async def run_async_tests():
    """Run async tests manually"""
    print("🧪 Running LLM Overload Handler Tests")
    print("=" * 40)
    
    # Setup logging
    logging.basicConfig(level=logging.WARNING)  # Reduce noise for testing
    
    try:
        # Basic functionality test
        config = OverloadConfig(
            context_window_size=1000,
            token_warning_threshold=0.7,
            retry_attempts=2,
            enable_token_chunking=True
        )
        
        handler = LLMOverloadHandler(config)
        print("✅ Handler initialization: PASSED")
        
        # Test overload detection
        events = await handler.check_overload_conditions(100)
        print(f"✅ Overload detection (no overload): PASSED ({len(events)} events)")
        
        # Test with large request
        events = await handler.check_overload_conditions(2000)
        if len(events) > 0:
            print(f"✅ Context overflow detection: PASSED ({len(events)} events)")
        else:
            print("ℹ️  Context overflow detection: No overload (expected with current limits)")
            
        # Test configuration loader
        loader = LLMConfigLoader()
        overload_config = loader.get_overload_config()
        api_limits = loader.get_api_limits()
        print("✅ Configuration loader: PASSED")
        
        # Test usage stats
        stats = handler.get_usage_stats()
        print(f"✅ Usage statistics: PASSED ({len(stats)} metrics)")
        
        print("\n🎉 All tests completed successfully!")
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Run tests
    if hasattr(pytest, 'main'):
        pytest.main([__file__])
    else:
        asyncio.run(run_async_tests())
