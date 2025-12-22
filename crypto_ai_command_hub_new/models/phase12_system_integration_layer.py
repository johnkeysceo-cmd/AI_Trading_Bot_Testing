"""Phase 12.5: System Integration Layer
Module orchestration and cross-phase data flow"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Dict, List, Any, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class IntegrationMetrics:
    iteration: int; orchestration_loss: float; module_efficiency: float; data_flow_throughput: float
    def to_dict(self):
        return {'iteration': self.iteration, 'orchestration_loss': float(self.orchestration_loss),
                'module_efficiency': float(self.module_efficiency), 'data_flow_throughput': float(self.data_flow_throughput)}

class ModuleRegistry:
    def __init__(self):
        self.modules = {}
        self.dependencies = {}
        self.data_cache = {}
    
    def register_module(self, name: str, module: Any, inputs: List[str] = None, outputs: List[str] = None):
        if inputs is None:
            inputs = []
        if outputs is None:
            outputs = []
        
        self.modules[name] = module
        self.dependencies[name] = inputs
        logger.info(f"Registered module: {name} (inputs: {inputs}, outputs: {outputs})")
    
    def get_module(self, name: str) -> Optional[Any]:
        return self.modules.get(name)
    
    def get_dependencies(self, name: str) -> List[str]:
        return self.dependencies.get(name, [])

class DataBus:
    def __init__(self):
        self.messages = {}
        self.subscribers = {}
    
    def publish(self, topic: str, data: Any):
        self.messages[topic] = data
        self._notify_subscribers(topic, data)
    
    def subscribe(self, topic: str, callback):
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)
    
    def _notify_subscribers(self, topic: str, data: Any):
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                callback(data)
    
    def get_data(self, topic: str) -> Optional[Any]:
        return self.messages.get(topic)

class Orchestrator:
    def __init__(self, registry: ModuleRegistry, bus: DataBus):
        self.registry = registry
        self.bus = bus
        self.execution_order = []
        self.module_outputs = {}
        self.compute_graph = {}
    
    def build_execution_plan(self) -> List[str]:
        # Topological sort of modules based on dependencies
        visited = set()
        order = []
        
        def visit(module_name: str):
            if module_name in visited:
                return
            visited.add(module_name)
            
            deps = self.registry.get_dependencies(module_name)
            for dep in deps:
                if dep in self.registry.modules:
                    visit(dep)
            
            order.append(module_name)
        
        for module_name in self.registry.modules:
            visit(module_name)
        
        self.execution_order = order
        logger.info(f"Execution plan: {' -> '.join(order)}")
        return order
    
    def execute_pipeline(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        # Publish input data
        for key, value in input_data.items():
            self.bus.publish(key, value)
        
        results = {}
        
        for module_name in self.execution_order:
            module = self.registry.get_module(module_name)
            deps = self.registry.get_dependencies(module_name)
            
            # Gather input from bus
            module_inputs = {}
            for dep in deps:
                data = self.bus.get_data(dep)
                if data is not None:
                    module_inputs[dep] = data
            
            # Execute module
            if hasattr(module, 'forward'):
                if module_inputs:
                    module_output = module.forward(**module_inputs)
                else:
                    module_output = module.forward(None)
            elif callable(module):
                module_output = module(**module_inputs) if module_inputs else module()
            else:
                module_output = module
            
            # Store and publish output
            self.module_outputs[module_name] = module_output
            self.bus.publish(module_name, module_output)
            results[module_name] = module_output
        
        return results

class SystemIntegrationLayer:
    def __init__(self, num_modules: int = 8, num_iterations: int = 100):
        self.num_modules = num_modules
        self.num_iterations = num_iterations
        
        self.registry = ModuleRegistry()
        self.bus = DataBus()
        self.orchestrator = Orchestrator(self.registry, self.bus)
        
        self.metrics_history = []
        self._setup_modules()
    
    def _setup_modules(self):
        # Create module placeholders (simulating 12 phases)
        module_names = [f'phase_{i}' for i in range(1, self.num_modules + 1)]
        
        for i, name in enumerate(module_names):
            # Create simple module
            module = self._create_phase_module(name, i)
            
            # Register with dependencies
            inputs = [] if i == 0 else [module_names[i-1]]
            self.registry.register_module(name, module, inputs=inputs, outputs=[name])
    
    def _create_phase_module(self, name: str, idx: int):
        class PhaseModule:
            def __init__(self, phase_name: str):
                self.phase_name = phase_name
                self.params = np.random.normal(0, 0.1, (16, 16))
            
            def forward(self, x=None):
                if x is None:
                    x = np.random.normal(0, 1, (8, 16))
                elif isinstance(x, dict):
                    x = np.random.normal(0, 1, (8, 16))
                
                # Transform data
                output = x @ self.params
                output = np.tanh(output)
                return output
        
        return PhaseModule(name)
    
    def train(self) -> dict:
        logger.info(f"Training System Integration: {self.num_iterations} iterations")
        
        # Build execution plan
        self.orchestrator.build_execution_plan()
        
        for iteration in range(self.num_iterations):
            # Generate input data
            input_data = {
                'raw_features': np.random.normal(0, 1, (8, 16))
            }
            
            # Execute pipeline
            pipeline_results = self.orchestrator.execute_pipeline(input_data)
            
            # Compute integration metrics
            orchestration_loss = 0
            module_efficiency = 0
            data_flow_throughput = 0
            
            for module_name, output in pipeline_results.items():
                if output is not None:
                    if isinstance(output, np.ndarray):
                        loss = np.mean(output ** 2)
                        orchestration_loss += loss
                        
                        efficiency = np.sum(np.abs(output) > 0.1) / output.size
                        module_efficiency += efficiency
                        
                        throughput = output.nbytes / 1024  # KB
                        data_flow_throughput += throughput
            
            num_modules = len(pipeline_results)
            if num_modules > 0:
                orchestration_loss /= num_modules
                module_efficiency /= num_modules
                data_flow_throughput /= num_modules
            
            self.metrics_history.append(
                IntegrationMetrics(iteration + 1, orchestration_loss, module_efficiency, 
                                 data_flow_throughput).to_dict()
            )
            
            if (iteration + 1) % 25 == 0:
                logger.info(f"Iteration {iteration + 1}: Orchestration Loss={orchestration_loss:.4f}, "
                          f"Module Efficiency={module_efficiency:.4f}, Throughput={data_flow_throughput:.2f} KB/s")
        
        return {
            'algorithm': 'System-Integration-Layer',
            'num_modules': self.num_modules,
            'num_iterations': self.num_iterations,
            'final_orchestration_loss': float(self.metrics_history[-1]['orchestration_loss']),
            'final_module_efficiency': float(self.metrics_history[-1]['module_efficiency']),
            'final_data_throughput': float(self.metrics_history[-1]['data_flow_throughput']),
            'avg_orchestration_loss': float(np.mean([m['orchestration_loss'] for m in self.metrics_history])),
            'avg_module_efficiency': float(np.mean([m['module_efficiency'] for m in self.metrics_history])),
            'avg_data_throughput': float(np.mean([m['data_flow_throughput'] for m in self.metrics_history])),
            'execution_order': self.orchestrator.execution_order,
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.5: System Integration Layer")
    logger.info("=" * 60)
    
    integration = SystemIntegrationLayer(num_modules=8, num_iterations=100)
    results = integration.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Execution order: {' -> '.join(results['execution_order'])}")
    logger.info(f"Final orchestration loss: {results['final_orchestration_loss']:.4f}")
    logger.info(f"Final module efficiency: {results['final_module_efficiency']:.4f}")
    logger.info(f"Final data throughput: {results['final_data_throughput']:.2f} KB/s")
    logger.info(f"Average orchestration loss: {results['avg_orchestration_loss']:.4f}")
    
    with open('phase12_system_integration_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_system_integration_results.json")
