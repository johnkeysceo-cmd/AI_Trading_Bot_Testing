"""Phase 12.4: Graph Attention Networks (GAT)
Node-wise attention and graph structure learning"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Tuple, List, Dict

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GATMetrics:
    epoch: int; attention_loss: float; node_classification_loss: float; node_accuracy: float
    def to_dict(self):
        return {'epoch': self.epoch, 'attention_loss': float(self.attention_loss),
                'node_classification_loss': float(self.node_classification_loss),
                'node_accuracy': float(self.node_accuracy)}

class GraphAttentionHead:
    def __init__(self, in_features: int = 16, out_features: int = 16, num_nodes: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.num_nodes = num_nodes
        
        self.W = np.random.normal(0, 0.1, (in_features, out_features))
        self.a = np.random.normal(0, 0.1, (out_features * 2,))
    
    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        # node_features: [num_nodes, in_features]
        # adj_matrix: [num_nodes, num_nodes] (binary adjacency)
        
        # Project features
        h = node_features @ self.W  # [num_nodes, out_features]
        
        # Compute attention scores
        num_nodes = h.shape[0]
        attention_scores = np.zeros((num_nodes, num_nodes))
        
        for i in range(num_nodes):
            for j in range(num_nodes):
                # Concatenate neighbor features
                features_concat = np.concatenate([h[i], h[j]])
                attention_scores[i, j] = np.dot(self.a, features_concat)
        
        # Apply mask (only attend to neighbors)
        attention_scores = np.where(adj_matrix > 0, attention_scores, -1e9)
        
        # Softmax over neighbors
        attention_weights = np.exp(attention_scores - np.max(attention_scores, axis=1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=1, keepdims=True + np.eye(num_nodes) * 1e-8)
        
        # Aggregate neighbors
        output = attention_weights @ h
        
        return output, attention_weights
    
    def backward(self, dL_doutput: np.ndarray, attention_weights: np.ndarray) -> np.ndarray:
        # Backprop through attention
        dL_dW = dL_doutput * 0.01
        return dL_dW

class MultiHeadGraphAttention:
    def __init__(self, in_features: int = 16, out_features: int = 16, num_heads: int = 4, num_nodes: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        
        self.heads = [GraphAttentionHead(in_features, out_features, num_nodes) for _ in range(num_heads)]
    
    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        head_outputs = []
        head_attentions = []
        
        for head in self.heads:
            output, attention = head.forward(node_features, adj_matrix)
            head_outputs.append(output)
            head_attentions.append(attention)
        
        # Concatenate head outputs
        combined_output = np.concatenate(head_outputs, axis=1)
        
        return combined_output, head_attentions

class GATLayer:
    def __init__(self, in_features: int = 16, out_features: int = 16, num_heads: int = 4, num_nodes: int = 10):
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.mha = MultiHeadGraphAttention(in_features, out_features, num_heads, num_nodes)
    
    def forward(self, node_features: np.ndarray, adj_matrix: np.ndarray) -> Tuple[np.ndarray, List[np.ndarray]]:
        output, attention_weights = self.mha.forward(node_features, adj_matrix)
        
        # Apply ELU activation
        output = np.where(output > 0, output, 0.1 * (np.exp(output) - 1))
        
        return output, attention_weights

class GraphAttentionNetwork:
    def __init__(self, num_nodes: int = 10, in_features: int = 16, hidden_features: int = 16,
                 num_classes: int = 4, num_heads: int = 4, num_layers: int = 2, num_epochs: int = 100):
        self.num_nodes = num_nodes
        self.in_features = in_features
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        
        self.layers = [GATLayer(in_features if i == 0 else hidden_features * num_heads,
                               hidden_features, num_heads, num_nodes) for i in range(num_layers)]
        
        # Classification head
        self.fc = np.random.normal(0, 0.1, (hidden_features * num_heads, num_classes))
        self.metrics_history = []
    
    def train(self, node_features: np.ndarray = None, adj_matrix: np.ndarray = None,
              labels: np.ndarray = None) -> dict:
        if node_features is None:
            node_features = np.random.normal(0, 1, (self.num_nodes, self.in_features))
        
        if adj_matrix is None:
            # Generate random graph
            adj_matrix = np.random.binomial(1, 0.3, (self.num_nodes, self.num_nodes))
            adj_matrix = np.maximum(adj_matrix, adj_matrix.T)  # Make symmetric
            np.fill_diagonal(adj_matrix, 1)  # Self-loops
        
        if labels is None:
            labels = np.random.randint(0, self.num_classes, self.num_nodes)
        
        logger.info(f"Training GAT: {self.num_epochs} epochs on {self.num_nodes} nodes")
        
        for epoch in range(self.num_epochs):
            # Forward pass
            x = node_features
            all_attentions = []
            
            for layer in self.layers:
                x, attention = layer.forward(x, adj_matrix)
                all_attentions.extend(attention)
            
            # Classification
            logits = x @ self.fc
            
            # Compute losses
            attention_loss = 0
            for attn in all_attentions:
                entropy = -np.sum(attn * np.log(attn + 1e-8))
                attention_loss += entropy
            attention_loss /= len(all_attentions)
            
            # Node classification loss
            classification_loss = 0
            correct = 0
            
            for i in range(self.num_nodes):
                pred = np.argmax(logits[i])
                loss = np.mean((logits[i] - np.eye(self.num_classes)[labels[i]]) ** 2)
                classification_loss += loss
                
                if pred == labels[i]:
                    correct += 1
                
                # Gradient update
                grad = (logits[i] - np.eye(self.num_classes)[labels[i]]).reshape(-1, 1)
                self.fc -= 0.001 * np.outer(x[i], grad.flatten())
            
            avg_attention_loss = float(attention_loss)
            avg_class_loss = float(classification_loss / self.num_nodes)
            accuracy = float(correct / self.num_nodes)
            
            self.metrics_history.append(
                GATMetrics(epoch + 1, avg_attention_loss, avg_class_loss, accuracy).to_dict()
            )
            
            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch + 1}: Attention Loss={avg_attention_loss:.4f}, "
                          f"Class Loss={avg_class_loss:.4f}, Accuracy={accuracy*100:.1f}%")
        
        return {
            'algorithm': 'Graph-Attention-Network',
            'num_nodes': self.num_nodes,
            'in_features': self.in_features,
            'num_classes': self.num_classes,
            'num_epochs': self.num_epochs,
            'final_attention_loss': float(self.metrics_history[-1]['attention_loss']),
            'final_class_loss': float(self.metrics_history[-1]['node_classification_loss']),
            'final_accuracy': float(self.metrics_history[-1]['node_accuracy']),
            'avg_accuracy': float(np.mean([m['node_accuracy'] for m in self.metrics_history])),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.4: Graph Attention Networks (GAT)")
    logger.info("=" * 60)
    
    gat = GraphAttentionNetwork(num_nodes=10, in_features=16, hidden_features=16,
                               num_classes=4, num_heads=4, num_layers=2, num_epochs=100)
    results = gat.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final attention loss: {results['final_attention_loss']:.4f}")
    logger.info(f"Final classification loss: {results['final_class_loss']:.4f}")
    logger.info(f"Final accuracy: {results['final_accuracy']*100:.1f}%")
    logger.info(f"Average accuracy: {results['avg_accuracy']*100:.1f}%")
    
    with open('phase12_graph_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_graph_attention_results.json")
