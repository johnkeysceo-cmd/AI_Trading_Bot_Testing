"""Phase 12.6: Real-Time Processing Engine
Streaming data processing with latency optimization"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Dict, List, Any, Callable, Optional
from collections import deque
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class RTMetrics:
    timestamp: float; latency_ms: float; throughput: float; queue_depth: int; error_rate: float
    def to_dict(self):
        return {'timestamp': float(self.timestamp), 'latency_ms': float(self.latency_ms),
                'throughput': float(self.throughput), 'queue_depth': self.queue_depth,
                'error_rate': float(self.error_rate)}

class StreamBuffer:
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        self.write_idx = 0
        self.read_idx = 0
    
    def put(self, item: Any):
        self.buffer.append(item)
        self.write_idx += 1
    
    def get(self) -> Optional[Any]:
        if len(self.buffer) > 0:
            item = self.buffer.popleft()
            self.read_idx += 1
            return item
        return None
    
    def size(self) -> int:
        return len(self.buffer)
    
    def is_full(self) -> bool:
        return len(self.buffer) >= self.max_size

class StreamProcessor:
    def __init__(self, processor_id: int, processing_fn: Callable = None):
        self.processor_id = processor_id
        self.processing_fn = processing_fn or (lambda x: x)
        self.processed_count = 0
        self.error_count = 0
        self.total_latency = 0
    
    def process(self, item: Any) -> Optional[Any]:
        try:
            start = time.time()
            
            result = self.processing_fn(item)
            
            latency = (time.time() - start) * 1000  # Convert to ms
            self.total_latency += latency
            self.processed_count += 1
            
            return result, latency
        except Exception as e:
            self.error_count += 1
            logger.warning(f"Processor {self.processor_id} error: {str(e)}")
            return None, 0
    
    def get_avg_latency(self) -> float:
        if self.processed_count == 0:
            return 0
        return self.total_latency / self.processed_count
    
    def get_error_rate(self) -> float:
        total = self.processed_count + self.error_count
        if total == 0:
            return 0
        return self.error_count / total

class StreamAggregator:
    def __init__(self, window_size: int = 100, aggregation_fn: Callable = None):
        self.window_size = window_size
        self.window = deque(maxlen=window_size)
        self.aggregation_fn = aggregation_fn or (lambda x: np.mean(x))
    
    def add(self, item: Any):
        self.window.append(item)
    
    def aggregate(self) -> Any:
        if len(self.window) == 0:
            return None
        
        items = list(self.window)
        if isinstance(items[0], np.ndarray):
            return np.mean(items, axis=0)
        elif isinstance(items[0], (int, float)):
            return self.aggregation_fn(items)
        else:
            return items[-1]

class RealtimeProcessingEngine:
    def __init__(self, num_processors: int = 4, buffer_size: int = 1000, 
                 aggregation_window: int = 100, num_iterations: int = 1000):
        self.num_processors = num_processors
        self.buffer_size = buffer_size
        self.aggregation_window = aggregation_window
        self.num_iterations = num_iterations
        
        # Core components
        self.input_buffer = StreamBuffer(buffer_size)
        self.output_buffer = StreamBuffer(buffer_size)
        
        # Processors
        self.processors = [
            StreamProcessor(i, self._create_processor_fn(i))
            for i in range(num_processors)
        ]
        
        # Aggregators
        self.aggregators = [
            StreamAggregator(aggregation_window)
            for _ in range(num_processors)
        ]
        
        self.metrics_history = []
        self.stream_data = []
    
    def _create_processor_fn(self, processor_id: int) -> Callable:
        def process_fn(data: Any) -> Any:
            if isinstance(data, np.ndarray):
                # Simulate ML inference
                result = np.sin(data) * np.cos(processor_id * 0.1)
            else:
                result = data * (processor_id + 1)
            
            return result
        
        return process_fn
    
    def ingest_data(self, data: Any):
        self.input_buffer.put(data)
    
    def process_batch(self) -> Dict[str, Any]:
        batch_results = []
        latencies = []
        error_count = 0
        
        while True:
            item = self.input_buffer.get()
            if item is None:
                break
            
            # Round-robin assignment to processors
            processor_idx = len(batch_results) % self.num_processors
            processor = self.processors[processor_idx]
            
            result, latency = processor.process(item)
            
            if result is not None:
                batch_results.append(result)
                latencies.append(latency)
                
                # Aggregate
                self.aggregators[processor_idx].add(result)
                
                # Output
                self.output_buffer.put(result)
            else:
                error_count += 1
        
        metrics = {
            'batch_size': len(batch_results),
            'latencies': latencies,
            'error_count': error_count,
            'results': batch_results
        }
        
        return metrics
    
    def train(self) -> dict:
        logger.info(f"Training Real-Time Processing: {self.num_iterations} iterations")
        
        total_processed = 0
        total_errors = 0
        all_latencies = []
        
        for iteration in range(self.num_iterations):
            # Generate streaming data
            num_items = np.random.randint(10, 50)
            data_batch = [np.random.normal(0, 1, (16,)) for _ in range(num_items)]
            
            # Ingest data
            for item in data_batch:
                self.ingest_data(item)
            
            # Process batch
            batch_metrics = self.process_batch()
            
            total_processed += batch_metrics['batch_size']
            total_errors += batch_metrics['error_count']
            all_latencies.extend(batch_metrics['latencies'])
            
            # Compute iteration metrics
            if batch_metrics['latencies']:
                avg_latency = np.mean(batch_metrics['latencies'])
            else:
                avg_latency = 0
            
            throughput = batch_metrics['batch_size'] / (avg_latency / 1000 + 1e-8) if avg_latency > 0 else 0
            queue_depth = self.input_buffer.size() + self.output_buffer.size()
            
            # Aggregate processor error rates
            total_processor_errors = sum(p.error_count for p in self.processors)
            total_processor_processed = sum(p.processed_count for p in self.processors)
            error_rate = total_processor_errors / (total_processor_processed + 1e-8)
            
            timestamp = time.time()
            self.metrics_history.append(
                RTMetrics(timestamp, avg_latency, throughput, queue_depth, error_rate).to_dict()
            )
            
            if (iteration + 1) % 200 == 0:
                logger.info(f"Iteration {iteration + 1}: Latency={avg_latency:.2f}ms, "
                          f"Throughput={throughput:.0f} items/s, Queue={queue_depth}, "
                          f"Error Rate={error_rate:.4f}")
        
        # Compute final statistics
        avg_latency_all = np.mean(all_latencies) if all_latencies else 0
        p95_latency = np.percentile(all_latencies, 95) if all_latencies else 0
        p99_latency = np.percentile(all_latencies, 99) if all_latencies else 0
        avg_throughput = np.mean([m['throughput'] for m in self.metrics_history])
        avg_error_rate = np.mean([m['error_rate'] for m in self.metrics_history])
        
        return {
            'algorithm': 'Realtime-Processing-Engine',
            'num_processors': self.num_processors,
            'buffer_size': self.buffer_size,
            'aggregation_window': self.aggregation_window,
            'num_iterations': self.num_iterations,
            'total_processed': total_processed,
            'total_errors': total_errors,
            'avg_latency_ms': float(avg_latency_all),
            'p95_latency_ms': float(p95_latency),
            'p99_latency_ms': float(p99_latency),
            'avg_throughput_items_per_sec': float(avg_throughput),
            'avg_error_rate': float(avg_error_rate),
            'processor_stats': [
                {
                    'processor_id': p.processor_id,
                    'processed_count': p.processed_count,
                    'error_count': p.error_count,
                    'avg_latency_ms': float(p.get_avg_latency())
                }
                for p in self.processors
            ],
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.6: Real-Time Processing Engine")
    logger.info("=" * 60)
    
    engine = RealtimeProcessingEngine(num_processors=4, buffer_size=1000, 
                                     aggregation_window=100, num_iterations=1000)
    results = engine.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Total items processed: {results['total_processed']}")
    logger.info(f"Total errors: {results['total_errors']}")
    logger.info(f"Average latency: {results['avg_latency_ms']:.2f}ms")
    logger.info(f"P95 latency: {results['p95_latency_ms']:.2f}ms")
    logger.info(f"P99 latency: {results['p99_latency_ms']:.2f}ms")
    logger.info(f"Average throughput: {results['avg_throughput_items_per_sec']:.0f} items/s")
    logger.info(f"Average error rate: {results['avg_error_rate']:.4f}")
    
    with open('phase12_realtime_processing_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_realtime_processing_results.json")
