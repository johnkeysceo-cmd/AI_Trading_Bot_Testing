"""Phase 12.3: Vision Transformers (ViT)
Image tokenization and transformer-based vision models"""

import numpy as np, json, logging
from dataclasses import dataclass
from typing import Tuple, List

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ViTMetrics:
    epoch: int; patch_loss: float; classification_loss: float; accuracy: float
    def to_dict(self):
        return {'epoch': self.epoch, 'patch_loss': float(self.patch_loss),
                'classification_loss': float(self.classification_loss), 'accuracy': float(self.accuracy)}

class PatchEmbedding:
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3, embed_dim: int = 768):
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_channels = in_channels
        self.embed_dim = embed_dim
        
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        self.proj = np.random.normal(0, 0.1, (self.patch_dim, embed_dim))
        self.patches = []
    
    def extract_patches(self, img: np.ndarray) -> np.ndarray:
        # img: [img_size, img_size, in_channels]
        h, w, c = img.shape
        patches = []
        
        for i in range(0, h, self.patch_size):
            for j in range(0, w, self.patch_size):
                patch = img[i:i+self.patch_size, j:j+self.patch_size, :]
                patch_flat = patch.reshape(-1)  # Flatten patch
                patches.append(patch_flat)
        
        patches = np.array(patches)
        return patches
    
    def forward(self, img: np.ndarray) -> np.ndarray:
        patches = self.extract_patches(img)
        patch_embeddings = patches @ self.proj
        
        self.patches = patches
        return patch_embeddings

class ClassToken:
    def __init__(self, embed_dim: int = 768):
        self.embed_dim = embed_dim
        self.cls_token = np.random.normal(0, 0.1, (1, embed_dim))
    
    def forward(self, patch_embeddings: np.ndarray) -> np.ndarray:
        batch_size, num_patches, embed_dim = patch_embeddings.shape if len(patch_embeddings.shape) == 3 else (1, patch_embeddings.shape[0], patch_embeddings.shape[1])
        
        cls_tokens = np.tile(self.cls_token, (batch_size if len(patch_embeddings.shape) == 3 else 1, 1))
        
        if len(patch_embeddings.shape) == 2:
            sequence = np.vstack([cls_tokens, patch_embeddings])
        else:
            sequence = np.concatenate([cls_tokens[:, np.newaxis, :], patch_embeddings], axis=1)
        
        return sequence

class PositionalEmbedding:
    def __init__(self, num_patches: int, embed_dim: int = 768):
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.pos_emb = np.random.normal(0, 0.1, (num_patches + 1, embed_dim))  # +1 for class token
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        return x + self.pos_emb

class TransformerBlock:
    def __init__(self, embed_dim: int = 768, num_heads: int = 12, ff_dim: int = 3072):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.ff_dim = ff_dim
        
        self.W_q = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_k = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_v = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        self.W_o = np.random.normal(0, 0.1, (embed_dim, embed_dim))
        
        self.W_ff1 = np.random.normal(0, 0.1, (embed_dim, ff_dim))
        self.W_ff2 = np.random.normal(0, 0.1, (ff_dim, embed_dim))
        self.b_ff1 = np.zeros(ff_dim)
        self.b_ff2 = np.zeros(embed_dim)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        # Multi-head self-attention
        Q = x @ self.W_q
        K = x @ self.W_k
        V = x @ self.W_v
        
        scores = np.dot(Q, K.T) / np.sqrt(self.embed_dim)
        attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
        
        attn_output = np.dot(attn_weights, V) @ self.W_o
        x = x + attn_output
        
        # Feed-forward
        ff_output = np.maximum(0, x @ self.W_ff1 + self.b_ff1)
        ff_output = ff_output @ self.W_ff2 + self.b_ff2
        x = x + ff_output
        
        return x

class VisionTransformer:
    def __init__(self, img_size: int = 224, patch_size: int = 16, in_channels: int = 3,
                 embed_dim: int = 768, num_heads: int = 12, num_layers: int = 12,
                 num_classes: int = 1000, num_epochs: int = 100):
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        
        self.patch_embedding = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.cls_token = ClassToken(embed_dim)
        self.pos_embedding = PositionalEmbedding(self.patch_embedding.num_patches, embed_dim)
        
        self.transformer_blocks = [TransformerBlock(embed_dim, num_heads) for _ in range(num_layers)]
        
        # Classification head
        self.fc = np.random.normal(0, 0.1, (embed_dim, num_classes))
        self.metrics_history = []
    
    def train(self, X: np.ndarray = None, y: np.ndarray = None) -> dict:
        if X is None:
            # Generate synthetic image batch: [batch_size=16, img_size=224, img_size=224, 3]
            X = np.random.uniform(0, 255, (16, self.img_size, self.img_size, 3)) / 255.0
            y = np.random.randint(0, self.num_classes, 16)
        
        batch_size = X.shape[0]
        logger.info(f"Training Vision Transformer: {self.num_epochs} epochs on {batch_size} images")
        
        for epoch in range(self.num_epochs):
            epoch_patch_loss = 0
            epoch_class_loss = 0
            correct = 0
            
            for i in range(batch_size):
                img = X[i]
                label = y[i] if y is not None else 0
                
                # Patch embedding
                patch_embeddings = self.patch_embedding.forward(img)
                patch_loss = np.mean((patch_embeddings - np.random.normal(0, 1, patch_embeddings.shape)) ** 2)
                epoch_patch_loss += patch_loss
                
                # Add class token
                sequence = self.cls_token.forward(patch_embeddings)
                
                # Add positional embedding
                sequence = self.pos_embedding.forward(sequence)
                
                # Transformer blocks
                for block in self.transformer_blocks:
                    sequence = block.forward(sequence)
                
                # Classification from class token
                cls_output = sequence[0] if len(sequence.shape) > 1 else sequence
                logits = cls_output @ self.fc
                
                # Loss and accuracy
                pred = np.argmax(logits)
                classification_loss = np.mean((logits - np.eye(self.num_classes)[label]) ** 2)
                epoch_class_loss += classification_loss
                
                if pred == label:
                    correct += 1
                
                # Simple gradient updates
                grad = (logits - np.eye(self.num_classes)[label]).reshape(-1, 1)
                self.fc -= 0.001 * np.outer(cls_output, grad.flatten())
            
            avg_patch_loss = epoch_patch_loss / batch_size
            avg_class_loss = epoch_class_loss / batch_size
            accuracy = correct / batch_size
            
            self.metrics_history.append(
                ViTMetrics(epoch + 1, avg_patch_loss, avg_class_loss, accuracy).to_dict()
            )
            
            if (epoch + 1) % 25 == 0:
                logger.info(f"Epoch {epoch + 1}: Patch Loss={avg_patch_loss:.4f}, "
                          f"Class Loss={avg_class_loss:.4f}, Accuracy={accuracy*100:.1f}%")
        
        return {
            'algorithm': 'Vision-Transformer',
            'img_size': self.img_size,
            'patch_size': self.patch_size,
            'embed_dim': self.embed_dim,
            'num_classes': self.num_classes,
            'num_epochs': self.num_epochs,
            'final_patch_loss': float(self.metrics_history[-1]['patch_loss']),
            'final_class_loss': float(self.metrics_history[-1]['classification_loss']),
            'final_accuracy': float(self.metrics_history[-1]['accuracy']),
            'avg_accuracy': float(np.mean([m['accuracy'] for m in self.metrics_history])),
            'metrics': self.metrics_history
        }

if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("Phase 12.3: Vision Transformers (ViT)")
    logger.info("=" * 60)
    
    vit = VisionTransformer(img_size=224, patch_size=16, embed_dim=768, num_heads=12, num_layers=12,
                           num_classes=1000, num_epochs=100)
    results = vit.train()
    
    logger.info("\nFINAL RESULTS")
    logger.info(f"Final patch loss: {results['final_patch_loss']:.4f}")
    logger.info(f"Final classification loss: {results['final_class_loss']:.4f}")
    logger.info(f"Final accuracy: {results['final_accuracy']*100:.1f}%")
    logger.info(f"Average accuracy: {results['avg_accuracy']*100:.1f}%")
    
    with open('phase12_vision_transformer_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info("✓ Results saved to phase12_vision_transformer_results.json")
