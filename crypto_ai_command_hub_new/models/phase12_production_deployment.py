"""Phase 12.7: Production Deployment System
Containerization, scaling, and monitoring"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class DeploymentMetrics:
    timestamp: float; cpu_usage: float; memory_usage: float; request_latency: float; uptime_seconds: float
    def to_dict(self):
        return {'timestamp': float(self.timestamp), 'cpu_usage': float(self.cpu_usage),
                'memory_usage': float(self.memory_usage), 'request_latency': float(self.request_latency),
                'uptime_seconds': float(self.uptime_seconds)}

class Container:
    def __init__(self, container_id: str, model_id: str, resource_limit: float = 4.0):
        self.container_id = container_id
        self.model_id = model_id
        self.resource_limit = resource_limit
        
        self.cpu_usage = 0.0
        self.memory_usage = 0.0
        self.is_running = False
        self.created_at = time.time()
        self.requests_processed = 0
    
    def start(self):
        self.is_running = True
        self.created_at = time.time()
        logger.info(f"Container {self.container_id} started (model: {self.model_id})")
    
    def stop(self):
        self.is_running = False
        logger.info(f"Container {self.container_id} stopped")
    
    def process_request(self, request_data: Any) -> Any:
        if not self.is_running:
            raise RuntimeError(f"Container {self.container_id} is not running")
        
        # Simulate model inference
        if isinstance(request_data, np.ndarray):
            result = np.sin(request_data) * np.cos(request_data)
        else:
            result = request_data
        
        # Update resource usage (simulated)
        self.cpu_usage = np.random.uniform(10, 80)
        self.memory_usage = np.random.uniform(500, 2000)  # MB
        self.requests_processed += 1
        
        if self.cpu_usage > self.resource_limit * 100:
            logger.warning(f"Container {self.container_id} exceeded CPU limit")
        
        return result
    
    def get_health(self) -> Dict[str, Any]:
        uptime = time.time() - self.created_at if self.is_running else 0
        
        return {
            'container_id': self.container_id,
            'status': 'running' if self.is_running else 'stopped',
            'model_id': self.model_id,
            'cpu_usage': float(self.cpu_usage),
            'memory_usage': float(self.memory_usage),
            'requests_processed': self.requests_processed,
            'uptime_seconds': float(uptime)
        }

class HorizontalScaler:
    def __init__(self, min_containers: int = 2, max_containers: int = 10, 
                 cpu_threshold: float = 80.0, memory_threshold: float = 2500.0):
        self.min_containers = min_containers
        self.max_containers = max_containers
        self.cpu_threshold = cpu_threshold
        self.memory_threshold = memory_threshold
        
        self.containers = {}
        self.container_counter = 0
        self.scaling_events = []
    
    def add_container(self, model_id: str) -> Container:
        if len(self.containers) >= self.max_containers:
            logger.warning("Maximum containers reached, cannot scale further")
            return None
        
        container_id = f"container_{self.container_counter}"
        self.container_counter += 1
        
        container = Container(container_id, model_id)
        container.start()
        self.containers[container_id] = container
        
        self.scaling_events.append({
            'timestamp': time.time(),
            'event': 'scale_up',
            'container_id': container_id
        })
        
        logger.info(f"Scaled up: {container_id}")
        return container
    
    def remove_container(self, container_id: str):
        if container_id in self.containers:
            self.containers[container_id].stop()
            del self.containers[container_id]
            
            self.scaling_events.append({
                'timestamp': time.time(),
                'event': 'scale_down',
                'container_id': container_id
            })
            
            logger.info(f"Scaled down: {container_id}")
    
    def evaluate_scaling(self) -> Optional[str]:
        if len(self.containers) == 0:
            return None
        
        avg_cpu = np.mean([c.cpu_usage for c in self.containers.values()])
        avg_memory = np.mean([c.memory_usage for c in self.containers.values()])
        
        # Scale up if above threshold
        if (avg_cpu > self.cpu_threshold or avg_memory > self.memory_threshold) and len(self.containers) < self.max_containers:
            return 'scale_up'
        
        # Scale down if below threshold
        if avg_cpu < 20 and len(self.containers) > self.min_containers:
            return 'scale_down'
        
        return None
    
    def get_container_health(self) -> List[Dict[str, Any]]:
        return [c.get_health() for c in self.containers.values()]

class LoadBalancer:
    def __init__(self, scaler: HorizontalScaler):
        self.scaler = scaler
        self.request_count = 0
        self.last_selected_idx = 0
    
    def select_container(self) -> Optional[Container]:
        if not self.scaler.containers:
            return None
        
        containers = list(self.scaler.containers.values())
        
        # Round-robin load balancing
        self.last_selected_idx = self.request_count % len(containers)
        selected = containers[self.last_selected_idx]
        self.request_count += 1
        
        return selected
    
    def route_request(self, request_data: Any) -> Any:
        container = self.select_container()
        
        if container is None:
            raise RuntimeError("No containers available")
        
        result = container.process_request(request_data)
        return result

class ProductionDeploymentSystem:
    def __init__(self, model_id: str = 'trading_model_v1', initial_containers: int = 2, 
                 num_iterations: int = 1000):
        self.model_id = model_id
        self.num_iterations = num_iterations
        
        self.scaler = HorizontalScaler(min_containers=2, max_containers=10)
        self.load_balancer = LoadBalancer(self.scaler)
        
        # Initialize containers
        for _ in range(initial_containers):
            self.scaler.add_container(model_id)
        
        self.metrics_history = []
        self.start_time = time.time()
    
    def train(self) -> dict:
        logger.info(f"Production Deployment: {self.num_iterations} iterations, {self.model_id}")
        
        for iteration in range(self.num_iterations):
            # Generate request
            request_data = np.random.normal(0, 1, (16,))
            
            # Route request
            try:
                result = self.load_balancer.route_request(request_data)
                
                # Check scaling decision
                scaling_decision = self.scaler.evaluate_scaling()
                
                if scaling_decision == 'scale_up':
                    self.scaler.add_container(self.model_id)
                elif scaling_decision == 'scale_down':
                    # Remove least-loaded container
                    containers = list(self.scaler.containers.values())
                    if containers:
                        min_container = min(containers, key=lambda c: c.requests_processed)
                        self.scaler.remove_container(min_container.container_id)
                
                # Collect metrics
                container_health = self.scaler.get_container_health()
                
                if container_health:
                    avg_cpu = np.mean([h['cpu_usage'] for h in container_health])
                    avg_memory = np.mean([h['memory_usage'] for h in container_health])
                    latency = np.random.uniform(10, 100)  # ms
                else:
                    avg_cpu = 0
                    avg_memory = 0
                    latency = 0
                
                uptime = time.time() - self.start_time
                
                self.metrics_history.append(
                    DeploymentMetrics(time.time(), avg_cpu, avg_memory, latency, uptime).to_dict()
                )
                
                if (iteration + 1) % 200 == 0:
                    num_containers = len(self.scaler.containers)
                    logger.info(f"Iteration {iteration + 1}: Containers={num_containers}, "
                              f"CPU={avg_cpu:.1f}%, Memory={avg_memory:.0f}MB, "
                              f"Latency={latency:.1f}ms")
            
            except Exception as e:
                logger.error(f"Request processing error: {str(e)}")
        
        # Final statistics
        final_health = self.scaler.get_container_health()
        total_requests = sum(h['requests_processed'] for h in final_health)
        
        return {
            'algorithm': 'Production-Deployment-System',
            'model_id': self.model_id,
            'num_iterations': self.num_iterations,
            'final_num_containers': len(self.scaler.containers),
            'total_requests_processed': total_requests,
            'total_scaling_events': len(self.scaler.scaling_events),
            'avg_cpu_usage': float(np.mean([m['cpu_usage'] for m in self.metrics_history])),
            'avg_memory_usage': float(np.mean([m['memory_usage'] for m in self.metrics_history])),
            'avg_request_latency_ms': float(np.mean([m['request_latency'] for m in self.metrics_history])),
            'p95_latency_ms': float(np.percentile([m['request_latency'] for m in self.metrics_history], 95)),
            'p99_latency_ms': float(np.percentile([m['request_latency'] for m in self.metrics_history], 99)),
            'total_uptime_seconds': float(time.time() - self.start_time),
            'container_health': final_health,
            'scaling_events': self.scaler.scaling_events,
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.7: Production Deployment System")
    logger.info("=" * 60)
    
    system = ProductionDeploymentSystem(model_id='trading_model_v1', initial_containers=2, num_iterations=1000)
    results = system.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Model ID: {results['model_id']}")
    logger.info(f"Final containers: {results['final_num_containers']}")
    logger.info(f"Total requests processed: {results['total_requests_processed']}")
    logger.info(f"Total scaling events: {results['total_scaling_events']}")
    logger.info(f"Average CPU usage: {results['avg_cpu_usage']:.1f}%")
    logger.info(f"Average memory usage: {results['avg_memory_usage']:.0f}MB")
    logger.info(f"Average latency: {results['avg_request_latency_ms']:.1f}ms")
    logger.info(f"P99 latency: {results['p99_latency_ms']:.1f}ms")
    
    with open('phase12_production_deployment_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_production_deployment_results.json")
