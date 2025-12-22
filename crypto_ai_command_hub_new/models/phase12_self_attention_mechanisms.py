"""Phase 12.1: Self-Attention Mechanisms
Multi-head scaled dot-product attention for sequence modeling"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class AttentionMetrics:
    epoch: int; attention_loss: float; avg_attention_entropy: float; learned_heads: int
    def to_dict(self):
        return {'epoch': self.epoch, 'attention_loss': float(self.attention_loss),
                'avg_attention_entropy': float(self.avg_attention_entropy), 'learned_heads': self.learned_heads}

class ScaledDotProductAttention:
    def __init__(self, embed_dim: int = 64, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.scale = np.sqrt(embed_dim)
        self.attention_weights = None
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray, 
                mask: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray]:
        # Scaled dot-product: softmax(Q*K^T / sqrt(d)) * V
        seq_len = Q.shape[0]
        
        # Compute attention scores
        scores = np.dot(Q, K.T) / self.scale
        
        if mask is not None:
            scores = np.where(mask, scores, -1e9)
        
        # Apply softmax
        self.attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        self.attention_weights = self.attention_weights / np.sum(self.attention_weights, axis=-1, keepdims=True)
        
        # Apply dropout
        if np.random.random() < self.dropout:
            self.attention_weights *= 1 / (1 - self.dropout)
        
        # Compute output
        output = np.dot(self.attention_weights, V)
        
        return output, self.attention_weights
    
    def backward(self, dL_doutput: np.ndarray, V: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        dL_dweights = np.dot(dL_doutput, V.T)
        dL_dV = np.dot(self.attention_weights.T, dL_doutput)
        
        return dL_dweights, dL_dV

class MultiHeadAttention:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        assert embed_dim % num_heads == 0, f"embed_dim={embed_dim} not divisible by num_heads={num_heads}"
        
        self.W_q = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_o = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        
        self.heads = [ScaledDotProductAttention(self.head_dim) for _ in range(num_heads)]
    
    def split_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len, embed_dim = x.shape
        x = x.reshape(seq_len, self.num_heads, self.head_dim)
        return x  # [seq_len, num_heads, head_dim]
    
    def combine_heads(self, x: np.ndarray) -> np.ndarray:
        seq_len, num_heads, head_dim = x.shape
        x = x.reshape(seq_len, num_heads * head_dim)
        return x  # [seq_len, embed_dim]
    
    def forward(self, Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                mask: np.ndarray = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Project inputs
        Q = Q @ self.W_q
        K = K @ self.W_k
        V = V @ self.W_v
        
        # Split heads
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)
        
        # Apply attention for each head
        head_outputs = []
        head_attentions = []
        
        for h in range(self.num_heads):
            output, attention = self.heads[h].forward(Q[:, h, :], K[:, h, :], V[:, h, :], mask)
            head_outputs.append(output)
            head_attentions.append(attention)
        
        # Combine heads
        head_outputs = np.stack(head_outputs, axis=1)  # [seq_len, num_heads, head_dim]
        output = self.combine_heads(head_outputs)
        
        # Final linear projection
        output = output @ self.W_o
        
        return output, head_attentions

class AttentionLayer:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, ff_dim: int = 1024, dropout: float = 0.1):
        self.embed_dim = embed_dim
        self.mha = MultiHeadAttention(embed_dim, num_heads)
        
        # Feed-forward network
        self.W1 = np.random.normal(0, 0.1, (embed_dim, ff_dim))
        self.W2 = np.random.normal(0, 0.1, (ff_dim, embed_dim))
        self.b1 = np.zeros(ff_dim)
        self.b2 = np.zeros(embed_dim)
        
        self.dropout = dropout
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> Tuple[np.ndarray, List[np.ndarray]]:
        # Multi-head attention
        attn_output, head_attentions = self.mha.forward(x, x, x, mask)
        attn_output = attn_output + x  # Residual connection
        
        # Feed-forward with ReLU
        ff_output = np.maximum(0, attn_output @ self.W1 + self.b1)
        ff_output = ff_output @ self.W2 + self.b2
        ff_output = ff_output + attn_output  # Residual connection
        
        return ff_output, head_attentions

class SelfAttentionTransformer:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4, num_epochs: int = 100):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_epochs = num_epochs
        
        self.layers = [AttentionLayer(embed_dim, num_heads) for _ in range(num_layers)]
        self.metrics_history = []
    
    def train(self, X: np.ndarray = None) -> dict:
        if X is None:
            # Generate synthetic sequence data: [batch_size=32, seq_len=10, embed_dim=256]
            X = np.random.normal(0, 1, (32, 10, self.embed_dim))
        
        batch_size, seq_len, embed_dim = X.shape
        logger.info(f"Training Self-Attention: {self.num_epochs} epochs on {batch_size} sequences")
        
        for epoch in range(self.num_epochs):
            epoch_loss = 0
            head_entropies = []
            active_heads = 0
            
            for batch_idx in range(batch_size):
                x_seq = X[batch_idx]  # [seq_len, embed_dim]
                
                # Forward pass through layers
                x = x_seq
                all_attentions = []
                
                for layer in self.layers:
                    x, head_attn = layer.forward(x)
                    all_attentions.append(head_attn)
                
                # Reconstruction loss
                loss = np.mean((x - x_seq) ** 2)
                epoch_loss += loss
                
                # Compute attention entropy
                for head_attns in all_attentions:
                    for attn_matrix in head_attns:
                        # Entropy of attention distribution
                        entropy = -np.sum(attn_matrix * np.log(attn_matrix + 1e-8))
                        head_entropies.append(entropy)
                        
                        if entropy > 0.5:
                            active_heads += 1
            
            avg_loss = epoch_loss / batch_size
            avg_entropy = np.mean(head_entropies) if head_entropies else 0
            
            self.metrics_history.append(
                AttentionMetrics(epoch + 1, avg_loss, avg_entropy, active_heads).to_dict()
            )
            
            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch + 1}: Loss={avg_loss:.4f}, "
                          f"Avg Entropy={avg_entropy:.4f}, Active Heads={active_heads}/{self.num_heads * self.num_layers}")
        
        return {
            'algorithm': 'Self-Attention-Transformer',
            'embed_dim': self.embed_dim,
            'num_heads': self.num_heads,
            'num_layers': self.num_layers,
            'num_epochs': self.num_epochs,
            'final_loss': float(self.metrics_history[-1]['attention_loss']),
            'avg_loss': float(np.mean([m['attention_loss'] for m in self.metrics_history])),
            'avg_attention_entropy': float(np.mean([m['avg_attention_entropy'] for m in self.metrics_history])),
            'max_active_heads': int(np.max([m['learned_heads'] for m in self.metrics_history])),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.1: Self-Attention Mechanisms")
    logger.info("=" * 60)
    
    transformer = SelfAttentionTransformer(embed_dim=256, num_heads=8, num_layers=4, num_epochs=100)
    results = transformer.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final loss: {results['final_loss']:.4f}")
    logger.info(f"Average loss: {results['avg_loss']:.4f}")
    logger.info(f"Average attention entropy: {results['avg_attention_entropy']:.4f}")
    logger.info(f"Max active heads: {results['max_active_heads']}/{results['num_heads'] * results['num_layers']}")
    
    with open('phase12_self_attention_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_self_attention_results.json")
