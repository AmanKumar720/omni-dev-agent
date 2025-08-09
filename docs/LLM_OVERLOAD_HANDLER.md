# LLM Overload Handler Documentation

## Overview

The LLM Overload Handler is a comprehensive system for managing and preventing overload conditions when working with Large Language Model APIs. It provides automatic detection, mitigation, and recovery from various overload scenarios including token limits, rate limits, context overflow, and concurrent request limits.

## Features

### 🔍 **Overload Detection**
- **Token Limit Monitoring**: Tracks token usage per minute/hour against API limits
- **Rate Limit Tracking**: Monitors request frequency to prevent API rate limiting
- **Context Window Management**: Prevents requests from exceeding model context limits
- **Concurrent Request Control**: Manages parallel API requests to avoid throttling
- **Circuit Breaker Pattern**: Automatically stops requests during persistent failures

### 🛠️ **Mitigation Strategies**
- **Automatic Token Chunking**: Splits large requests into manageable pieces
- **Context Truncation**: Intelligently reduces request size while preserving meaning
- **Exponential Backoff**: Implements smart retry logic with increasing delays
- **Request Queuing**: Manages request flow to stay within limits
- **Priority Handling**: Processes critical requests first during overload

### 📊 **Monitoring & Analytics**
- **Real-time Usage Metrics**: Track token usage, request rates, and system health
- **Overload Event Logging**: Detailed logs of all overload conditions and responses
- **Performance Statistics**: Historical data on system performance and reliability
- **Cost Tracking**: Monitor API usage costs and budget compliance

## Quick Start

### Basic Usage

```python
from src.error_handling import LLMOverloadHandler, OverloadConfig

# Create handler with default configuration
handler = LLMOverloadHandler()

# Your LLM API function
async def call_llm_api(prompt: str) -> dict:
    # Your actual API call here
    return await your_llm_client.chat_completion(prompt=prompt)

# Check for overload before making request
events = await handler.check_overload_conditions(
    request_tokens=len(prompt.split()) * 1.3,  # Rough token estimate
    context={'service': 'vision_analysis'}
)

# Apply mitigation if needed and make request
result = await handler.apply_mitigation_strategy(
    events, 
    call_llm_api, 
    prompt=prompt
)
```

### Using the Decorator

```python
from src.error_handling import with_llm_overload_protection

@with_llm_overload_protection()
async def analyze_scene(prompt: str) -> str:
    """Automatically protected LLM call"""
    response = await your_llm_client.chat_completion(
        prompt=prompt,
        max_tokens=500
    )
    return response['choices'][0]['message']['content']

# Use normally - overload protection is automatic
result = await analyze_scene("Describe this image...")
```

### Configuration-Based Setup

```python
from src.error_handling import create_configured_handler

# Create handler with service-specific configuration
handler = create_configured_handler(
    service_name='vision_analysis',
    environment='production',
    api_provider='gpt4'
)

# Handler is now configured specifically for vision analysis in production
```

## Configuration

### Configuration File Structure

The system uses YAML configuration files located at `src/config/llm_overload_config.yaml`:

```yaml
# Service-specific configuration
services:
  vision_analysis:
    context_window_size: 8192
    max_context_utilization: 0.85
    token_warning_threshold: 0.7
    token_critical_threshold: 0.9
    retry_attempts: 3
    enable_token_chunking: true
    chunk_overlap: 200

# Environment-specific settings
environments:
  production:
    token_warning_threshold: 0.6
    token_critical_threshold: 0.8
    retry_attempts: 3
    circuit_breaker_threshold: 3

# API provider limits
api_limits:
  gpt4:
    requests_per_minute: 500
    tokens_per_minute: 10000
    max_tokens_per_request: 8192
    max_concurrent_requests: 10
```

### Environment Variables

Override configuration using environment variables:

```bash
export LLM_CONTEXT_WINDOW_SIZE=16384
export LLM_TOKEN_WARNING_THRESHOLD=0.7
export LLM_ENABLE_TOKEN_CHUNKING=true
```

## Integration Examples

### Vision Analysis Service

```python
from src.error_handling import create_configured_handler

class VisionAnalysisService:
    def __init__(self):
        self.overload_handler = create_configured_handler(
            service_name='vision_analysis',
            environment='production'
        )
    
    async def analyze_scene(self, objects: list, context: str) -> str:
        prompt = self._build_analysis_prompt(objects, context)
        
        # Estimate tokens
        estimated_tokens = len(prompt.split()) * 1.3
        
        # Check for overload
        events = await self.overload_handler.check_overload_conditions(
            estimated_tokens,
            {'scene_objects': len(objects)}
        )
        
        # Apply mitigation and make request
        result = await self.overload_handler.apply_mitigation_strategy(
            events,
            self._call_llm,
            prompt=prompt
        )
        
        return self._extract_response(result)
```

### Batch Processing

```python
async def process_batch_with_overload_protection(requests: list):
    handler = create_configured_handler('batch_processing')
    results = []
    
    for i, request in enumerate(requests):
        # Check overload for each request
        events = await handler.check_overload_conditions(
            request['estimated_tokens'],
            {'batch_index': i, 'total': len(requests)}
        )
        
        if events:
            logger.info(f"Overload detected for request {i}, applying mitigation")
        
        # Process with protection
        result = await handler.apply_mitigation_strategy(
            events,
            make_llm_request,
            **request
        )
        
        results.append(result)
        
        # Brief pause between requests
        await asyncio.sleep(0.1)
    
    return results
```

### Real-time Processing

```python
# Configuration for real-time processing
realtime_config = OverloadConfig(
    context_window_size=2048,
    max_context_utilization=0.7,
    retry_attempts=1,  # Minimal retries for speed
    enable_token_chunking=False,  # No chunking for real-time
    base_retry_delay=0.1,
    max_retry_delay=2.0
)

handler = LLMOverloadHandler(realtime_config)

async def process_realtime_request(prompt: str):
    """Fast processing with minimal overload handling"""
    events = await handler.check_overload_conditions(
        len(prompt.split()) * 1.3
    )
    
    if events and any(e.severity == OverloadSeverity.CRITICAL for e in events):
        # Fail fast for real-time processing
        raise OverloadError("System overloaded, try again later")
    
    return await handler.apply_mitigation_strategy(
        events, 
        make_fast_llm_call, 
        prompt=prompt
    )
```

## Monitoring and Alerting

### Usage Statistics

```python
# Get current usage statistics
stats = handler.get_usage_stats()

print(f"Token utilization: {stats['token_utilization_minute']:.2%}")
print(f"Request utilization: {stats['request_utilization_minute']:.2%}")
print(f"Active requests: {stats['active_requests']}")
print(f"Recent overload events: {stats['recent_overload_events']}")
```

### System Health Check

```python
async def system_health_check():
    """Comprehensive system health assessment"""
    handler = create_configured_handler()
    
    # Get usage statistics
    stats = handler.get_usage_stats()
    
    health = {
        'status': 'healthy',
        'issues': []
    }
    
    # Check token utilization
    if stats['token_utilization_minute'] > 0.8:
        health['status'] = 'warning'
        health['issues'].append('High token utilization')
    
    # Check request utilization  
    if stats['request_utilization_minute'] > 0.9:
        health['status'] = 'critical'
        health['issues'].append('High request utilization')
    
    # Check recent overload events
    if stats['recent_overload_events'] > 5:
        health['status'] = 'warning'
        health['issues'].append('Frequent overload events')
    
    return health
```

### Custom Alerts

```python
class OverloadAlerting:
    def __init__(self, handler: LLMOverloadHandler):
        self.handler = handler
        self.alert_thresholds = {
            'token_utilization': 0.8,
            'request_utilization': 0.9,
            'overload_frequency': 5
        }
    
    async def check_and_alert(self):
        """Check conditions and send alerts if needed"""
        stats = self.handler.get_usage_stats()
        
        # Check token utilization
        if stats['token_utilization_minute'] > self.alert_thresholds['token_utilization']:
            await self._send_alert(
                'High Token Utilization',
                f"Token usage at {stats['token_utilization_minute']:.2%}"
            )
        
        # Check request utilization
        if stats['request_utilization_minute'] > self.alert_thresholds['request_utilization']:
            await self._send_alert(
                'High Request Utilization', 
                f"Request usage at {stats['request_utilization_minute']:.2%}"
            )
    
    async def _send_alert(self, title: str, message: str):
        """Send alert notification"""
        logger.warning(f"ALERT: {title} - {message}")
        # Add your alerting logic here (email, Slack, etc.)
```

## Best Practices

### 1. Choose Appropriate Configuration

```python
# For high-throughput services
high_throughput_config = OverloadConfig(
    token_warning_threshold=0.6,  # Conservative threshold
    retry_attempts=2,             # Fewer retries for speed
    enable_token_chunking=True,   # Handle large requests
    base_retry_delay=0.5         # Quick retries
)

# For accuracy-critical services  
accuracy_config = OverloadConfig(
    token_warning_threshold=0.9,  # Allow higher utilization
    retry_attempts=5,             # More retries for reliability
    max_retry_delay=120.0,        # Longer delays acceptable
    enable_token_chunking=False   # Avoid context fragmentation
)
```

### 2. Implement Graceful Degradation

```python
async def analyze_with_fallback(prompt: str):
    """Implement fallback strategies during overload"""
    try:
        # Try full analysis
        return await full_analysis_with_protection(prompt)
    except OverloadError as e:
        if e.overload_type == OverloadType.CONTEXT_OVERFLOW:
            # Fallback to simpler analysis
            simplified_prompt = simplify_prompt(prompt)
            return await simple_analysis(simplified_prompt)
        elif e.overload_type == OverloadType.RATE_LIMIT:
            # Return cached result if available
            cached = get_cached_result(prompt)
            if cached:
                return cached
            # Wait and retry once
            await asyncio.sleep(60)
            return await simple_analysis(prompt)
        else:
            raise
```

### 3. Monitor and Optimize

```python
# Regular monitoring
async def monitoring_loop():
    while True:
        health = await system_health_check()
        if health['status'] != 'healthy':
            logger.warning(f"System health: {health}")
            
        # Log usage statistics every 5 minutes
        stats = handler.get_usage_stats()
        logger.info(f"Usage: {stats['token_utilization_minute']:.2%} tokens, "
                   f"{stats['request_utilization_minute']:.2%} requests")
        
        await asyncio.sleep(300)  # 5 minutes
```

### 4. Test Overload Conditions

```python
async def test_overload_handling():
    """Test system behavior under overload conditions"""
    handler = LLMOverloadHandler()
    
    # Simulate token limit overload
    large_prompt = "word " * 10000  # Very large prompt
    
    events = await handler.check_overload_conditions(
        len(large_prompt.split()) * 1.3
    )
    
    assert len(events) > 0, "Should detect overload"
    assert any(e.overload_type == OverloadType.CONTEXT_OVERFLOW for e in events)
    
    # Test mitigation
    result = await handler.apply_mitigation_strategy(
        events,
        mock_llm_call,
        prompt=large_prompt
    )
    
    # Should chunk the request
    assert isinstance(result, list), "Should return chunked results"
```

## API Reference

### LLMOverloadHandler

#### Methods

- `__init__(config: OverloadConfig)`: Initialize handler
- `check_overload_conditions(tokens: int, context: dict) -> List[OverloadEvent]`: Check for overload
- `apply_mitigation_strategy(events: List[OverloadEvent], func: Callable, *args, **kwargs) -> Any`: Apply mitigation
- `get_usage_stats() -> Dict[str, Any]`: Get current usage statistics
- `reset_statistics()`: Reset all usage statistics

### OverloadConfig

#### Parameters

- `context_window_size: int`: Maximum context window size
- `max_context_utilization: float`: Maximum percentage of context to use (0.0-1.0)  
- `token_warning_threshold: float`: Warning threshold for token usage (0.0-1.0)
- `token_critical_threshold: float`: Critical threshold for token usage (0.0-1.0)
- `retry_attempts: int`: Number of retry attempts
- `base_retry_delay: float`: Base delay between retries (seconds)
- `max_retry_delay: float`: Maximum delay between retries (seconds)
- `enable_token_chunking: bool`: Enable automatic token chunking
- `chunk_overlap: int`: Number of tokens to overlap between chunks

### OverloadType Enum

- `TOKEN_LIMIT`: Token usage limit exceeded
- `RATE_LIMIT`: Request rate limit exceeded  
- `CONTEXT_OVERFLOW`: Context window overflow
- `CONCURRENT_REQUESTS`: Too many concurrent requests
- `MEMORY_PRESSURE`: System memory pressure
- `API_QUOTA`: API quota exceeded

### OverloadSeverity Enum

- `LOW`: Minor overload, automatic handling
- `MEDIUM`: Moderate overload, may cause delays
- `HIGH`: Significant overload, mitigation required
- `CRITICAL`: Severe overload, immediate action needed

## Troubleshooting

### Common Issues

#### 1. High Token Utilization

**Symptoms**: Token utilization consistently above 80%
**Solutions**:
- Reduce `max_context_utilization` in configuration
- Enable token chunking for large requests
- Implement request queuing
- Consider upgrading API plan

#### 2. Frequent Rate Limit Errors  

**Symptoms**: Many rate limit overload events
**Solutions**:
- Increase `base_retry_delay` 
- Implement exponential backoff
- Reduce concurrent requests
- Spread requests over time

#### 3. Context Overflow Issues

**Symptoms**: Requests failing due to context overflow
**Solutions**:
- Enable token chunking
- Implement context truncation
- Break large requests into smaller parts
- Use appropriate context window size

#### 4. Poor Performance

**Symptoms**: Slow response times, many retries
**Solutions**:
- Tune retry configuration
- Implement caching
- Use appropriate service configuration
- Monitor and optimize token usage

### Debug Logging

```python
# Enable detailed logging
import logging
logging.getLogger('src.error_handling.llm_overload_handler').setLevel(logging.DEBUG)

# Custom log handler for overload events
class OverloadLogHandler:
    def __init__(self, handler: LLMOverloadHandler):
        self.handler = handler
        
    def log_overload_event(self, event: OverloadEvent):
        logger.warning(
            f"Overload detected: {event.overload_type.value} "
            f"(severity: {event.severity.value}) - {event.context}"
        )
```

## Performance Considerations

### Memory Usage

The handler maintains in-memory queues for tracking usage:
- `minute_tokens`: Up to 1000 entries
- `hour_tokens`: Up to 10000 entries  
- `overload_events`: All events (consider periodic cleanup)

### CPU Impact

- Minimal overhead for normal operations
- Text chunking can be CPU intensive for very large texts
- Statistics calculation scales with queue sizes

### Network Impact

- Retry logic may increase total API calls
- Chunking increases number of requests but reduces individual request size
- Exponential backoff helps reduce network pressure during overload

## Future Enhancements

### Planned Features

1. **Machine Learning Optimization**: Learn from usage patterns to optimize thresholds
2. **Multi-Provider Support**: Automatically switch between API providers
3. **Advanced Load Balancing**: Distribute requests across multiple endpoints
4. **Predictive Overload Detection**: Predict overload before it occurs
5. **Integration with Monitoring Tools**: Prometheus, Grafana, DataDog support
6. **Cost Optimization**: Automatic cost-aware request routing

### Contributing

To contribute to the LLM Overload Handler:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality  
4. Update documentation
5. Submit a pull request

### License

This feature is part of the Omni-Dev Agent project and follows the same license terms.
