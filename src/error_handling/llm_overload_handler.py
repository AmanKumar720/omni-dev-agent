"""
LLM Overload Handler for Omni-Dev Agent
Manages LLM API limits, token usage, and prevents overload conditions.
"""

import asyncio
import time
import logging
from typing import Dict, Any, Optional, List, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import json
from pathlib import Path
import threading
from collections import deque, defaultdict
import hashlib

from components.llm_manager import LLMManager # Import the LLMManager


class OverloadType(Enum):
    """Types of LLM overload conditions"""
    TOKEN_LIMIT = "token_limit"
    RATE_LIMIT = "rate_limit"
    CONTEXT_OVERFLOW = "context_overflow" 
    MEMORY_PRESSURE = "memory_pressure"
    CONCURRENT_REQUESTS = "concurrent_requests"
    API_QUOTA = "api_quota"


class OverloadSeverity(Enum):
    """Severity levels for overload conditions"""
    LOW = "low"
    MEDIUM = "medium" 
    HIGH = "high"
    CRITICAL = "critical"


class RetryStrategy(Enum):
    """Retry strategies for different overload types"""
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    IMMEDIATE_RETRY = "immediate_retry"
    CIRCUIT_BREAKER = "circuit_breaker"
    TOKEN_CHUNKING = "token_chunking"


@dataclass
class OverloadEvent:
    """Represents an overload event"""
    timestamp: datetime
    overload_type: OverloadType
    severity: OverloadSeverity
    context: Dict[str, Any]
    recovery_time: Optional[datetime] = None
    mitigation_applied: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TokenUsage:
    """Tracks token usage for requests"""
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class APILimits:
    """API rate limits and quotas"""
    requests_per_minute: int = 60
    tokens_per_minute: int = 90000
    requests_per_hour: int = 3600
    tokens_per_hour: int = 1000000
    max_tokens_per_request: int = 4096
    max_concurrent_requests: int = 10
    daily_quota: int = 1000000


@dataclass 
class OverloadConfig:
    """Configuration for overload handling"""
    token_warning_threshold: float = 0.8  # Warn at 80% of limit
    token_critical_threshold: float = 0.95  # Critical at 95% of limit
    context_window_size: int = 8192
    max_context_utilization: float = 0.9  # Use max 90% of context window
    retry_attempts: int = 3
    base_retry_delay: float = 1.0
    max_retry_delay: float = 60.0
    circuit_breaker_threshold: int = 5  # Failures before opening circuit
    circuit_breaker_timeout: int = 300  # 5 minutes
    enable_token_chunking: bool = True
    chunk_overlap: int = 100  # Token overlap between chunks
    

class LLMOverloadHandler:
    """
    Main handler for LLM overload conditions.
    Monitors usage, detects overload, and applies mitigation strategies.
    """
    
    def __init__(self, llm_manager: LLMManager, config: Optional[OverloadConfig] = None):
        self.llm_manager = llm_manager # LLMManager instance
        self.config = config or OverloadConfig()
        self.api_limits = APILimits()
        
        # Monitoring data
        self.token_usage_history: deque = deque(maxlen=1000)
        self.request_history: deque = deque(maxlen=1000)
        self.overload_events: List[OverloadEvent] = []
        
        # Rate limiting tracking
        self.minute_requests = deque(maxlen=100)
        self.minute_tokens = deque(maxlen=1000)
        self.hour_requests = deque(maxlen=3600)
        self.hour_tokens = deque(maxlen=10000)
        
        # Circuit breaker state
        self.circuit_breaker_failures = defaultdict(int)
        self.circuit_breaker_last_failure = defaultdict(datetime)
        
        # Concurrency control
        self.active_requests = 0
        self.request_semaphore = asyncio.Semaphore(self.api_limits.max_concurrent_requests)
        self._lock = threading.Lock()
        
        # Logging
        self.logger = logging.getLogger(__name__ + ".LLMOverloadHandler")
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging for overload handler"""
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)
        self.logger.setLevel(logging.INFO)
        
    async def check_overload_conditions(self, 
                                       request_tokens: int,
                                       context: Optional[Dict[str, Any]] = None) -> List[OverloadEvent]:
        """
        Check for current overload conditions before making a request
        
        Args:
            request_tokens: Number of tokens for the planned request
            context: Additional context information
            
        Returns:
            List of current overload events
        """
        events = []
        current_time = datetime.now()
        context = context or {}
        
        # Check token limits
        if self._check_token_limits(request_tokens):
            severity = self._get_token_limit_severity(request_tokens)
            events.append(OverloadEvent(
                timestamp=current_time,
                overload_type=OverloadType.TOKEN_LIMIT,
                severity=severity,
                context={"request_tokens": request_tokens, **context}
            ))
            
        # Check rate limits
        if self._check_rate_limits():
            events.append(OverloadEvent(
                timestamp=current_time,
                overload_type=OverloadType.RATE_LIMIT,
                severity=OverloadSeverity.HIGH,
                context=context
            ))
            
        # Check context overflow
        if self._check_context_overflow(request_tokens):
            events.append(OverloadEvent(
                timestamp=current_time,
                overload_type=OverloadType.CONTEXT_OVERFLOW,
                severity=OverloadSeverity.MEDIUM,
                context={"request_tokens": request_tokens, **context}
            ))
            
        # Check concurrent request limits
        if self._check_concurrent_limits():
            events.append(OverloadEvent(
                timestamp=current_time,
                overload_type=OverloadType.CONCURRENT_REQUESTS,
                severity=OverloadSeverity.HIGH,
                context=context
            ))
            
        # Store events
        self.overload_events.extend(events)
        
        return events
        
    def _check_token_limits(self, request_tokens: int) -> bool:
        """Check if token limits would be exceeded"""
        current_time = datetime.now()
        
        # Clean old entries
        self._cleanup_old_entries(self.minute_tokens, timedelta(minutes=1))
        self._cleanup_old_entries(self.hour_tokens, timedelta(hours=1))
        
        minute_usage = sum(usage.total_tokens for _, usage in self.minute_tokens)
        hour_usage = sum(usage.total_tokens for _, usage in self.hour_tokens)
        
        return (minute_usage + request_tokens > self.api_limits.tokens_per_minute or
                hour_usage + request_tokens > self.api_limits.tokens_per_hour or
                request_tokens > self.api_limits.max_tokens_per_request)
                
    def _check_rate_limits(self) -> bool:
        """Check if rate limits would be exceeded"""
        current_time = datetime.now()
        
        # Clean old entries
        self._cleanup_old_entries(self.minute_requests, timedelta(minutes=1))
        self._cleanup_old_entries(self.hour_requests, timedelta(hours=1))
        
        return (len(self.minute_requests) >= self.api_limits.requests_per_minute or
                len(self.hour_requests) >= self.api_limits.requests_per_hour)
                
    def _check_context_overflow(self, request_tokens: int) -> bool:
        """Check if context window would overflow"""
        max_usable_tokens = int(self.config.context_window_size * self.config.max_context_utilization)
        return request_tokens > max_usable_tokens
        
    def _check_concurrent_limits(self) -> bool:
        """Check if concurrent request limits would be exceeded"""
        return self.active_requests >= self.api_limits.max_concurrent_requests
        
    def _get_token_limit_severity(self, request_tokens: int) -> OverloadSeverity:
        """Determine severity of token limit overload"""
        minute_usage = sum(usage.total_tokens for _, usage in self.minute_tokens)
        utilization = (minute_usage + request_tokens) / self.api_limits.tokens_per_minute
        
        if utilization >= 1.0:
            return OverloadSeverity.CRITICAL
        elif utilization >= self.config.token_critical_threshold:
            return OverloadSeverity.HIGH
        elif utilization >= self.config.token_warning_threshold:
            return OverloadSeverity.MEDIUM
        else:
            return OverloadSeverity.LOW
            
    def _cleanup_old_entries(self, entries: deque, max_age: timedelta):
        """Remove entries older than max_age"""
        current_time = datetime.now()
        while entries and current_time - entries[0][0] > max_age:
            entries.popleft()
            
    async def apply_mitigation_strategy(self, events: List[OverloadEvent],
                                       prompt: str,
                                       use_gemini: bool = True,
                                       use_ollama: bool = False,
                                       **kwargs) -> Any:
        """
        Apply appropriate mitigation strategies for overload events
        
        Args:
            events: List of overload events to mitigate
            prompt: The prompt for the LLM request
            use_gemini: Flag to indicate if Gemini should be used
            use_ollama: Flag to indicate if Ollama should be used
            **kwargs: Additional arguments for the LLM generate_content method
            
        Returns:
            Result of the request or error
        """
        if not events:
            return await self._execute_request(prompt, use_gemini, use_ollama, **kwargs)
            
        # Sort events by severity
        events.sort(key=lambda e: list(OverloadSeverity).index(e.severity), reverse=True)
        
        primary_event = events[0]
        
        # Apply mitigation based on primary overload type
        if primary_event.overload_type == OverloadType.TOKEN_LIMIT:
            return await self._handle_token_limit_overload(prompt, primary_event, use_gemini, use_ollama, **kwargs)
        elif primary_event.overload_type == OverloadType.RATE_LIMIT:
            return await self._handle_rate_limit_overload(prompt, primary_event, use_gemini, use_ollama, **kwargs)
        elif primary_event.overload_type == OverloadType.CONTEXT_OVERFLOW:
            return await self._handle_context_overflow(prompt, primary_event, use_gemini, use_ollama, **kwargs)
        elif primary_event.overload_type == OverloadType.CONCURRENT_REQUESTS:
            return await self._handle_concurrent_limit_overload(prompt, primary_event, use_gemini, use_ollama, **kwargs)
        else:
            return await self._handle_generic_overload(prompt, primary_event, use_gemini, use_ollama, **kwargs)
            
    async def _handle_token_limit_overload(self, prompt: str, event: OverloadEvent, 
                                          use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Handle token limit overload"""
        self.logger.warning(f"Token limit overload detected: {event.severity.value}")
        
        if event.severity in [OverloadSeverity.CRITICAL, OverloadSeverity.HIGH]:
            if self.config.enable_token_chunking:
                return await self._chunk_and_process(prompt, use_gemini, use_ollama, **kwargs)
            else:
                # Wait for token availability
                wait_time = self._calculate_token_wait_time()
                self.logger.info(f"Waiting {wait_time:.1f}s for token availability")
                await asyncio.sleep(wait_time)
                
        return await self._execute_with_retry(prompt, use_gemini, use_ollama, **kwargs)
        
    async def _handle_rate_limit_overload(self, prompt: str, event: OverloadEvent,
                                         use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Handle rate limit overload"""
        self.logger.warning("Rate limit overload detected")
        
        wait_time = self._calculate_rate_limit_wait_time()
        self.logger.info(f"Waiting {wait_time:.1f}s for rate limit reset")
        await asyncio.sleep(wait_time)
        
        return await self._execute_with_retry(prompt, use_gemini, use_ollama, **kwargs)
        
    async def _handle_context_overflow(self, prompt: str, event: OverloadEvent,
                                      use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Handle context window overflow"""
        self.logger.warning("Context overflow detected")
        
        prompt = self._truncate_context(prompt)
            
        return await self._execute_with_retry(prompt, use_gemini, use_ollama, **kwargs)
        
    async def _handle_concurrent_limit_overload(self, prompt: str, event: OverloadEvent,
                                               use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Handle concurrent request limit overload"""
        self.logger.warning("Concurrent request limit reached")
        
        async with self.request_semaphore:
            return await self._execute_with_retry(prompt, use_gemini, use_ollama, **kwargs)
            
    async def _handle_generic_overload(self, prompt: str, event: OverloadEvent,
                                      use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Handle generic overload with exponential backoff"""
        wait_time = self.config.base_retry_delay
        return await self._execute_with_backoff(prompt, wait_time, use_gemini, use_ollama, **kwargs)
        
    async def _chunk_and_process(self, prompt: str,
                                use_gemini: bool, use_ollama: bool, **kwargs) -> List[Any]:
        """Chunk large requests and process separately"""
        self.logger.info("Chunking large request into smaller parts")
        
        chunks = self._split_text_into_chunks(prompt)
        results = []
        
        for i, chunk in enumerate(chunks):
            self.logger.info(f"Processing chunk {i+1}/{len(chunks)}")
            
            result = await self._execute_with_retry(chunk, use_gemini, use_ollama, **kwargs)
            results.append(result)
            
            # Small delay between chunks
            if i < len(chunks) - 1:
                await asyncio.sleep(0.5)
                
        return results
        
    def _split_text_into_chunks(self, text: str) -> List[str]:
        """Split text into manageable chunks with overlap"""
        max_chunk_size = int(self.config.context_window_size * self.config.max_context_utilization)
        
        # Simple word-based chunking (could be enhanced with proper tokenization)
        words = text.split()
        chunks = []
        
        if len(words) <= max_chunk_size:
            return [text]
            
        i = 0
        while i < len(words):
            chunk_end = min(i + max_chunk_size, len(words))
            chunk_words = words[i:chunk_end]
            chunks.append(' '.join(chunk_words))
            
            # Move index back by overlap amount for next chunk
            i += max_chunk_size - self.config.chunk_overlap
            
        return chunks
        
    def _truncate_context(self, text: str) -> str:
        """Truncate context to fit within limits"""
        max_length = int(self.config.context_window_size * self.config.max_context_utilization)
        words = text.split()
        
        if len(words) <= max_length:
            return text
            
        truncated = words[:max_length]
        return ' '.join(truncated) + "... [truncated]"
        
    async def _execute_with_retry(self, prompt: str, use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Execute request with retry logic"""
        last_exception = None
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await self._execute_request(prompt, use_gemini, use_ollama, **kwargs)
            except Exception as e:
                last_exception = e
                self.logger.warning(f"Request attempt {attempt + 1} failed: {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    wait_time = self._calculate_retry_delay(attempt)
                    await asyncio.sleep(wait_time)
                    
        raise last_exception
        
    async def _execute_with_backoff(self, prompt: str, initial_delay: float,
                                   use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Execute request with exponential backoff"""
        delay = initial_delay
        
        for attempt in range(self.config.retry_attempts):
            try:
                return await self._execute_request(prompt, use_gemini, use_ollama, **kwargs)
            except Exception as e:
                self.logger.warning(f"Request failed (attempt {attempt + 1}): {e}")
                
                if attempt < self.config.retry_attempts - 1:
                    await asyncio.sleep(delay)
                    delay = min(delay * 2, self.config.max_retry_delay)
                else:
                    raise
                    
    async def _execute_request(self, prompt: str, use_gemini: bool, use_ollama: bool, **kwargs) -> Any:
        """Execute the actual request with monitoring"""
        start_time = time.time()
        
        with self._lock:
            self.active_requests += 1
            
        try:
            # Use LLMManager to generate content
            result = await self.llm_manager.generate_content(prompt, use_gemini=use_gemini, use_ollama=use_ollama, **kwargs)
            
            # Record successful request
            self._record_successful_request(time.time() - start_time, prompt)
            
            return result
            
        except Exception as e:
            # Record failed request
            self._record_failed_request(str(e))
            raise
            
        finally:
            with self._lock:
                self.active_requests -= 1
                
    def _calculate_token_wait_time(self) -> float:
        """Calculate how long to wait for token availability"""
        if not self.minute_tokens:
            return 0.0
            
        # Find when oldest token usage will expire
        oldest_time = self.minute_tokens[0][0]
        wait_time = 60 - (datetime.now() - oldest_time).total_seconds()
        return max(0, wait_time)
        
    def _calculate_rate_limit_wait_time(self) -> float:
        """Calculate how long to wait for rate limit reset"""
        if not self.minute_requests:
            return 0.0
            
        oldest_time = self.minute_requests[0]
        wait_time = 60 - (datetime.now() - oldest_time).total_seconds()
        return max(0, wait_time)
        
    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate retry delay with exponential backoff"""
        delay = self.config.base_retry_delay * (2 ** attempt)
        return min(delay, self.config.max_retry_delay)
        
    def _record_successful_request(self, duration: float, prompt: str):
        """Record successful request for monitoring"""
        current_time = datetime.now()
        
        # Estimate token usage (rough approximation)
        estimated_tokens = len(prompt.split()) * 1.3  # Average tokens per word
        
        token_usage = TokenUsage(
            prompt_tokens=int(estimated_tokens * 0.8),
            completion_tokens=int(estimated_tokens * 0.2),
            total_tokens=int(estimated_tokens)
        )
        
        self.token_usage_history.append((current_time, token_usage))
        self.minute_tokens.append((current_time, token_usage))
        self.hour_tokens.append((current_time, token_usage))
        self.minute_requests.append(current_time)
        self.hour_requests.append(current_time)
        
    def _record_failed_request(self, error: str):
        """Record failed request for circuit breaker logic"""
        current_time = datetime.now()
        error_hash = hashlib.md5(error.encode()).hexdigest()[:8]
        
        self.circuit_breaker_failures[error_hash] += 1
        self.circuit_breaker_last_failure[error_hash] = current_time
        
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        current_time = datetime.now()
        
        # Clean old entries
        self._cleanup_old_entries(self.minute_tokens, timedelta(minutes=1))
        self._cleanup_old_entries(self.hour_tokens, timedelta(hours=1))
        
        minute_tokens = sum(usage.total_tokens for _, usage in self.minute_tokens)
        hour_tokens = sum(usage.total_tokens for _, usage in self.hour_tokens)
        
        return {
            "current_time": current_time.isoformat(),
            "active_requests": self.active_requests,
            "minute_requests": len(self.minute_requests),
            "hour_requests": len(self.hour_requests), 
            "minute_tokens": minute_tokens,
            "hour_tokens": hour_tokens,
            "token_utilization_minute": minute_tokens / self.api_limits.tokens_per_minute,
            "request_utilization_minute": len(self.minute_requests) / self.api_limits.requests_per_minute,
            "recent_overload_events": len([e for e in self.overload_events 
                                         if (current_time - e.timestamp).total_seconds() < 3600])
        }
        
    def reset_statistics(self):
        """Reset all usage statistics"""
        self.token_usage_history.clear()
        self.request_history.clear()
        self.overload_events.clear()
        self.minute_requests.clear()
        self.minute_tokens.clear()
        self.hour_requests.clear()
        self.hour_tokens.clear()
        self.circuit_breaker_failures.clear()
        self.circuit_breaker_last_failure.clear()
        
        self.logger.info("Usage statistics reset")


# Decorator for automatic overload handling
def with_llm_overload_protection(handler: Optional[LLMOverloadHandler] = None,
                                 use_gemini: bool = True,
                                 use_ollama: bool = False):
    """Decorator to automatically handle LLM overload conditions"""
    
    def decorator(func):
        async def wrapper(*args, **kwargs):
            # Extract prompt from kwargs or args (assuming prompt is a keyword arg or first positional arg)
            prompt = kwargs.get('prompt', args[0] if args else '')
            
            # Estimate token count for request
            estimated_tokens = len(prompt.split()) * 1.3 if prompt else 100
            
            # If handler is not provided, create a default one (requires LLMManager to be passed later)
            # For simplicity in decorator usage, we'll assume LLMManager is globally accessible or passed implicitly
            # In a real application, you'd likely inject the LLMManager into the handler or the decorated class
            current_handler = handler
            if current_handler is None:
                # This part is tricky for a decorator without a global LLMManager instance
                # For now, we'll raise an error if no handler is provided, forcing explicit injection
                raise RuntimeError("LLMOverloadHandler instance must be provided to the decorator or initialized globally.")

            # Check for overload conditions
            events = await current_handler.check_overload_conditions(
                int(estimated_tokens),
                {'function': func.__name__}
            )
            
            # Apply mitigation if needed
            return await current_handler.apply_mitigation_strategy(events, prompt, use_gemini, use_ollama, **kwargs)
            
        return wrapper
    return decorator