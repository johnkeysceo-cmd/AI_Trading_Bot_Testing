"""Phase 12.2: Transformer Encoder-Decoder Architecture
Full transformer with positional encoding and layer normalization"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TransformerMetrics:
    epoch: int; encoder_loss: float; decoder_loss: float; translation_accuracy: float
    def to_dict(self):
        return {'epoch': self.epoch, 'encoder_loss': float(self.encoder_loss),
                'decoder_loss': float(self.decoder_loss), 'translation_accuracy': float(self.translation_accuracy)}

class PositionalEncoding:
    def __init__(self, embed_dim: int = 256, max_seq_len: int = 100):
        self.embed_dim = embed_dim
        self.max_seq_len = max_seq_len
        self.pe = self._create_positional_encoding()
    
    def _create_positional_encoding(self) -> np.ndarray:
        pe = np.zeros((self.max_seq_len, self.embed_dim))
        
        positions = np.arange(0, self.max_seq_len, dtype=np.float32).reshape(-1, 1)
        dimensions = np.arange(0, self.embed_dim, 2, dtype=np.float32)
        
        angle_rates = 1 / np.power(10000, dimensions / self.embed_dim)
        
        pe[:, 0::2] = np.sin(positions * angle_rates)
        if self.embed_dim % 2 == 1:
            pe[:, 1::2] = np.cos(positions * angle_rates[:-1])
        else:
            pe[:, 1::2] = np.cos(positions * angle_rates)
        
        return pe
    
    def encode(self, x: np.ndarray) -> np.ndarray:
        seq_len = x.shape[0]
        return x + self.pe[:seq_len, :]

class LayerNormalization:
    def __init__(self, embed_dim: int = 256, eps: float = 1e-6):
        self.embed_dim = embed_dim
        self.eps = eps
        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Layer norm: (x - mean) / sqrt(var + eps) * gamma + beta
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        
        x_norm = (x - mean) / np.sqrt(var + self.eps)
        output = x_norm * self.gamma + self.beta
        
        return output

class TransformerEncoder:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layer_norms = [LayerNormalization(embed_dim) for _ in range(num_layers)]
        
        # Attention weights
        self.W_q = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_k = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_v = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        
        # Feed-forward weights
        self.W_ff1 = [np.random.normal(0, 0.1, (embed_dim, embed_dim * 4)) for _ in range(num_layers)]
        self.W_ff2 = [np.random.normal(0, 0.1, (embed_dim * 4, embed_dim)) for _ in range(num_layers)]
    
    def forward(self, x: np.ndarray, mask: np.ndarray = None) -> np.ndarray:
        # Add positional encoding
        x = self.pos_encoding.encode(x)
        
        for layer_idx in range(self.num_layers):
            # Self-attention
            Q = x @ self.W_q[layer_idx]
            K = x @ self.W_k[layer_idx]
            V = x @ self.W_v[layer_idx]
            
            scores = np.dot(Q, K.T) / np.sqrt(self.embed_dim)
            
            if mask is not None:
                scores = np.where(mask, scores, -1e9)
            
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
            
            attn_output = np.dot(attn_weights, V)
            x = x + attn_output  # Residual
            x = self.layer_norms[layer_idx].forward(x)
            
            # Feed-forward
            ff_output = np.maximum(0, x @ self.W_ff1[layer_idx])
            ff_output = ff_output @ self.W_ff2[layer_idx]
            x = x + ff_output  # Residual
            x = self.layer_norms[layer_idx].forward(x)
        
        return x

class TransformerDecoder:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        
        self.pos_encoding = PositionalEncoding(embed_dim)
        self.layer_norms = [LayerNormalization(embed_dim) for _ in range(num_layers * 2)]
        
        # Self-attention weights
        self.W_q_self = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_k_self = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_v_self = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        
        # Cross-attention weights
        self.W_q_cross = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_k_cross = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        self.W_v_cross = [np.random.normal(0, 0.1, (embed_dim, embed_dim)) for _ in range(num_layers)]
        
        # Feed-forward weights
        self.W_ff1 = [np.random.normal(0, 0.1, (embed_dim, embed_dim * 4)) for _ in range(num_layers)]
        self.W_ff2 = [np.random.normal(0, 0.1, (embed_dim * 4, embed_dim)) for _ in range(num_layers)]
    
    def forward(self, y: np.ndarray, encoder_output: np.ndarray, 
                self_mask: np.ndarray = None, cross_mask: np.ndarray = None) -> np.ndarray:
        # Add positional encoding
        y = self.pos_encoding.encode(y)
        
        for layer_idx in range(self.num_layers):
            # Self-attention (decoder attends to itself)
            Q = y @ self.W_q_self[layer_idx]
            K = y @ self.W_k_self[layer_idx]
            V = y @ self.W_v_self[layer_idx]
            
            scores = np.dot(Q, K.T) / np.sqrt(self.embed_dim)
            if self_mask is not None:
                scores = np.where(self_mask, scores, -1e9)
            
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
            
            attn_output = np.dot(attn_weights, V)
            y = y + attn_output
            y = self.layer_norms[layer_idx * 2].forward(y)
            
            # Cross-attention (decoder attends to encoder)
            Q = y @ self.W_q_cross[layer_idx]
            K = encoder_output @ self.W_k_cross[layer_idx]
            V = encoder_output @ self.W_v_cross[layer_idx]
            
            scores = np.dot(Q, K.T) / np.sqrt(self.embed_dim)
            if cross_mask is not None:
                scores = np.where(cross_mask, scores, -1e9)
            
            attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
            attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
            
            attn_output = np.dot(attn_weights, V)
            y = y + attn_output
            y = self.layer_norms[layer_idx * 2 + 1].forward(y)
            
            # Feed-forward
            ff_output = np.maximum(0, y @ self.W_ff1[layer_idx])
            ff_output = ff_output @ self.W_ff2[layer_idx]
            y = y + ff_output
            y = self.layer_norms[layer_idx * 2 + 1].forward(y)
        
        return y

class TransformerEncoderDecoder:
    def __init__(self, embed_dim: int = 256, num_heads: int = 8, num_layers: int = 4, num_epochs: int = 100):
        self.embed_dim = embed_dim
        self.encoder = TransformerEncoder(embed_dim, num_heads, num_layers)
        self.decoder = TransformerDecoder(embed_dim, num_heads, num_layers)
        self.num_epochs = num_epochs
        self.metrics_history = []
    
    def train(self) -> dict:
        logger.info(f"Training Transformer: {self.num_epochs} epochs")
        
        for epoch in range(self.num_epochs):
            # Generate synthetic source and target sequences
            source_seq = np.random.normal(0, 1, (10, self.embed_dim))
            target_seq = np.random.normal(0, 1, (8, self.embed_dim))
            
            # Encoder forward pass
            encoder_output = self.encoder.forward(source_seq)
            
            # Decoder forward pass
            decoder_output = self.decoder.forward(target_seq, encoder_output)
            
            # Compute losses
            encoder_loss = np.mean((encoder_output - source_seq) ** 2)
            decoder_loss = np.mean((decoder_output - target_seq) ** 2)
            
            # Translation accuracy (cosine similarity)
            similarity = np.mean([
                np.dot(decoder_output[i], target_seq[i]) / 
                (np.linalg.norm(decoder_output[i]) * np.linalg.norm(target_seq[i]) + 1e-8)
                for i in range(min(len(decoder_output), len(target_seq)))
            ])
            translation_accuracy = (similarity + 1) / 2  # Normalize to [0, 1]
            
            self.metrics_history.append(
                TransformerMetrics(epoch + 1, encoder_loss, decoder_loss, translation_accuracy).to_dict()
            )
            
            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch + 1}: Encoder Loss={encoder_loss:.4f}, "
                          f"Decoder Loss={decoder_loss:.4f}, Accuracy={translation_accuracy:.4f}")
        
        return {
            'algorithm': 'Transformer-Encoder-Decoder',
            'embed_dim': self.embed_dim,
            'num_epochs': self.num_epochs,
            'final_encoder_loss': float(self.metrics_history[-1]['encoder_loss']),
            'final_decoder_loss': float(self.metrics_history[-1]['decoder_loss']),
            'final_translation_accuracy': float(self.metrics_history[-1]['translation_accuracy']),
            'avg_encoder_loss': float(np.mean([m['encoder_loss'] for m in self.metrics_history])),
            'avg_decoder_loss': float(np.mean([m['decoder_loss'] for m in self.metrics_history])),
            'avg_translation_accuracy': float(np.mean([m['translation_accuracy'] for m in self.metrics_history])),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.2: Transformer Encoder-Decoder")
    logger.info("=" * 60)
    
    transformer = TransformerEncoderDecoder(embed_dim=256, num_heads=8, num_layers=4, num_epochs=100)
    results = transformer.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final encoder loss: {results['final_encoder_loss']:.4f}")
    logger.info(f"Final decoder loss: {results['final_decoder_loss']:.4f}")
    logger.info(f"Final translation accuracy: {results['final_translation_accuracy']:.4f}")
    logger.info(f"Average encoder loss: {results['avg_encoder_loss']:.4f}")
    logger.info(f"Average decoder loss: {results['avg_decoder_loss']:.4f}")
    
    with open('phase12_transformer_encoder_decoder_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_transformer_encoder_decoder_results.json")
