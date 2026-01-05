"""
Brain-Grounded Language System

Implements language processing based on neuroscience principles:
- No tokenization (character/byte-level processing)
- lATL abstraction control (the "dimmer")
- Hebbian associative learning
- Wernicke (comprehension) / Broca (production) separation
- N400-like prediction error detection

Reference: docs/Bericht_Menschliches_Gehirn.docx
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Optional, List
import math


class CharacterPatternDetector(nn.Module):
    """
    Mimics Visual Word Form Area (VWFA) in visual cortex.
    Detects n-gram patterns: 'ing', 'tion', 'un-', etc.
    
    Key insight: Brain processes letter PATTERNS, not discrete tokens.
    """
    
    def __init__(self, d_pattern: int = 256):
        super().__init__()
        self.d_pattern = d_pattern
        
        # Byte-level input (0-255 for raw characters)
        self.byte_embed = nn.Embedding(256, 64, padding_idx=0)
        
        # Multiple receptive fields (like visual cortex hierarchy)
        # Different kernel sizes detect different pattern lengths
        self.bigram_conv = nn.Conv1d(64, d_pattern, kernel_size=2, padding=1)
        self.trigram_conv = nn.Conv1d(64, d_pattern, kernel_size=3, padding=1)
        self.fourgram_conv = nn.Conv1d(64, d_pattern, kernel_size=4, padding=2)
        self.fivegram_conv = nn.Conv1d(64, d_pattern, kernel_size=5, padding=2)
        
        # Combine all pattern detections
        self.combine = nn.Sequential(
            nn.Linear(d_pattern * 4, d_pattern * 2),
            nn.LayerNorm(d_pattern * 2),
            nn.GELU(),
            nn.Linear(d_pattern * 2, d_pattern)
        )
        
        # Initialize with small weights for stability
        self._init_weights()
        
    def _init_weights(self):
        for conv in [self.bigram_conv, self.trigram_conv, 
                     self.fourgram_conv, self.fivegram_conv]:
            nn.init.kaiming_normal_(conv.weight, mode='fan_out', nonlinearity='relu')
            nn.init.zeros_(conv.bias)
    
    def forward(self, byte_seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            byte_seq: [batch, seq_len] raw byte values (0-255)
        
        Returns:
            patterns: [batch, d_pattern] detected character patterns
        """
        # Embed bytes to dense vectors
        x = self.byte_embed(byte_seq)  # [batch, seq_len, 64]
        x = x.transpose(1, 2)  # [batch, 64, seq_len] for conv1d
        
        # Detect patterns at multiple scales
        p2 = F.gelu(self.bigram_conv(x))    # Detect 'th', 'in', 'er'
        p3 = F.gelu(self.trigram_conv(x))   # Detect 'ing', 'the', 'and'
        p4 = F.gelu(self.fourgram_conv(x))  # Detect 'tion', 'ness'
        p5 = F.gelu(self.fivegram_conv(x))  # Detect 'ation', 'iness'
        
        # Max pool to get most salient patterns (like winner-take-all in cortex)
        p2 = torch.max(p2, dim=2)[0]  # [batch, d_pattern]
        p3 = torch.max(p3, dim=2)[0]
        p4 = torch.max(p4, dim=2)[0]
        p5 = torch.max(p5, dim=2)[0]
        
        # Combine all pattern detections
        patterns = torch.cat([p2, p3, p4, p5], dim=1)  # [batch, d_pattern * 4]
        combined = self.combine(patterns)  # [batch, d_pattern]
        
        return combined


class SemanticAbstraction(nn.Module):
    """
    Left Anterior Temporal Lobe (lATL) - Abstraction control.
    Converts patterns → concepts at controllable abstraction level.
    
    The "dimmer" concept: Controls how abstract vs concrete the representation is.
    - Level 0.0: Very concrete (savant-like, sees all details)
    - Level 1.0: Very abstract (sees only categories)
    """
    
    def __init__(self, d_pattern: int = 256, d_semantic: int = 512, 
                 num_abstraction_levels: int = 5):
        super().__init__()
        self.d_semantic = d_semantic
        self.num_levels = num_abstraction_levels
        
        # Multi-level abstraction hierarchy (like cortical layers)
        self.abstraction_levels = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_pattern if i == 0 else d_semantic, d_semantic),
                nn.LayerNorm(d_semantic),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(d_semantic, d_semantic)
            )
            for i in range(num_abstraction_levels)
        ])
        
        # Learnable abstraction control parameter (the "dimmer")
        self.abstraction_control = nn.Parameter(torch.tensor(0.0))  # Sigmoid → 0.5
        
    def forward(self, patterns: torch.Tensor, 
                abstraction_level: Optional[float] = None) -> Tuple[torch.Tensor, float]:
        """
        Args:
            patterns: [batch, d_pattern] from character detector
            abstraction_level: None (use learned), or float 0-1 to control
        
        Returns:
            semantic: [batch, d_semantic] semantic representation
            level: actual abstraction level used
        """
        # Use explicit level or learned parameter
        if abstraction_level is None:
            level = torch.sigmoid(self.abstraction_control).item()
        else:
            level = float(abstraction_level)
        
        # Process through abstraction hierarchy
        representations = []
        x = patterns
        
        for i, layer in enumerate(self.abstraction_levels):
            x = layer(x)
            representations.append(x)
        
        # Mix representations based on abstraction level
        # Low level (0.0) = early layers (concrete details)
        # High level (1.0) = late layers (abstract categories)
        device = patterns.device
        num_layers = len(representations)
        
        # Gaussian weighting centered on desired level
        center = level * (num_layers - 1)
        weights = torch.zeros(num_layers, device=device)
        
        for i in range(num_layers):
            dist_sq = (i - center) ** 2
            weights[i] = np.exp(-dist_sq / 1.5)
        
        weights = weights / (weights.sum() + 1e-8)
        
        # Weighted combination of abstraction levels
        semantic = sum(w * rep for w, rep in zip(weights, representations))
        
        return semantic, level
    
    def set_abstraction(self, level: float):
        """Manually control abstraction level (for experiments)."""
        # Inverse sigmoid to set parameter
        level = max(0.01, min(0.99, level))  # Clamp for numerical stability
        logit = math.log(level / (1 - level))
        self.abstraction_control.data = torch.tensor(logit)


class AssociativeSemanticNetwork(nn.Module):
    """
    Cortical semantic network - concepts activate related concepts.
    Implements associative thinking: "apple" → red, sweet, fruit, tree, etc.
    
    Uses Hebbian learning: "Cells that fire together, wire together"
    """
    
    def __init__(self, d_semantic: int = 512, num_concepts: int = 1000):
        super().__init__()
        self.d_semantic = d_semantic
        self.num_concepts = num_concepts
        
        # Concept embeddings (like semantic memory nodes)
        self.concept_embeddings = nn.Parameter(
            torch.randn(num_concepts, d_semantic) * 0.02
        )
        
        # Hebbian association matrix (learned connections between concepts)
        # Initialized as weak identity (self-connections)
        self.associations = nn.Parameter(
            torch.eye(num_concepts) * 0.1 + torch.randn(num_concepts, num_concepts) * 0.01
        )
        
        # Projection back to continuous semantic space
        self.semantic_projection = nn.Linear(num_concepts, d_semantic)
        
        # Hebbian learning rate
        self.hebbian_lr = 0.001
        self.hebbian_decay = 0.01
        
    def forward(self, semantic_input: torch.Tensor, 
                num_activation_steps: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            semantic_input: [batch, d_semantic]
            num_activation_steps: How many spreading activation iterations
        
        Returns:
            activated_semantic: [batch, d_semantic] with associations activated
            concept_activation: [batch, num_concepts] activation pattern
        """
        # Match input to concept nodes (soft retrieval)
        # Cosine similarity to find relevant concepts
        input_norm = F.normalize(semantic_input, dim=-1)
        embed_norm = F.normalize(self.concept_embeddings, dim=-1)
        
        similarities = torch.matmul(input_norm, embed_norm.T)  # [batch, num_concepts]
        
        # Initial activation (soft attention over concepts)
        activation = torch.softmax(similarities * 5.0, dim=-1)  # Temperature scaling
        
        # Spreading activation through associative network
        assoc_weights = torch.sigmoid(self.associations)  # Positive connections
        
        for _ in range(num_activation_steps):
            # Hebbian spreading: concepts activate related concepts
            spread = torch.matmul(activation, assoc_weights)
            activation = activation + 0.3 * spread
            activation = torch.sigmoid(activation * 2 - 1)  # Normalize
        
        # Project activated concepts back to semantic space
        activated_semantic = self.semantic_projection(activation)
        
        return activated_semantic, activation
    
    def hebbian_update(self, activation_pattern: torch.Tensor):
        """
        Update associations using Hebbian rule.
        "Cells that fire together, wire together"
        
        Args:
            activation_pattern: [batch, num_concepts]
        """
        with torch.no_grad():
            # Compute co-activation across batch
            # [batch, num_concepts] → correlation matrix
            act_centered = activation_pattern - activation_pattern.mean(dim=0, keepdim=True)
            
            # Outer product gives co-activation
            co_activation = torch.matmul(act_centered.T, act_centered) / activation_pattern.shape[0]
            
            # Hebbian update with decay (prevents runaway growth)
            delta = self.hebbian_lr * co_activation - self.hebbian_decay * self.associations
            self.associations.add_(delta)
            
            # Keep associations bounded
            self.associations.data.clamp_(-2.0, 2.0)


class WernickeComprehension(nn.Module):
    """
    Wernicke's Area - Language comprehension pathway.
    Character patterns → Semantic understanding
    
    Integrates:
    - Pattern detection (VWFA)
    - Semantic abstraction (lATL)
    - Associative activation (cortex)
    - Prediction (for N400-like surprise)
    """
    
    def __init__(self, d_pattern: int = 256, d_semantic: int = 512,
                 num_concepts: int = 1000):
        super().__init__()
        
        self.pattern_detector = CharacterPatternDetector(d_pattern)
        self.abstraction = SemanticAbstraction(d_pattern, d_semantic)
        self.associative_network = AssociativeSemanticNetwork(d_semantic, num_concepts)
        
        # Prediction system (for N400-like surprise detection)
        self.predictor = nn.LSTM(d_semantic, d_semantic, num_layers=2, batch_first=True)
        self.prediction_head = nn.Linear(d_semantic, d_semantic)
        
    def forward(self, byte_sequence: torch.Tensor, 
                abstraction_level: Optional[float] = None,
                return_all: bool = False):
        """
        Args:
            byte_sequence: [batch, seq_len] raw bytes
            abstraction_level: Optional control over abstraction (0-1)
            return_all: If True, return intermediate representations
        
        Returns:
            semantic: [batch, d_semantic] understood meaning
            concept_activation: [batch, num_concepts] activated concepts
        """
        # 1. Detect character patterns (VWFA-like)
        patterns = self.pattern_detector(byte_sequence)
        
        # 2. Abstract to semantic level (lATL function)
        semantic, level = self.abstraction(patterns, abstraction_level)
        
        # 3. Activate associative network (cortical spreading)
        activated_semantic, concept_activation = self.associative_network(semantic)
        
        if return_all:
            return {
                'patterns': patterns,
                'semantic': semantic,
                'activated_semantic': activated_semantic,
                'concept_activation': concept_activation,
                'abstraction_level': level
            }
        
        return activated_semantic, concept_activation
    
    def compute_surprise(self, byte_sequence: torch.Tensor,
                        previous_context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute prediction error (N400-like response).
        High surprise = unexpected word given context.
        
        Args:
            byte_sequence: [batch, seq_len] current input
            previous_context: [batch, d_semantic] semantic context
        
        Returns:
            surprise: [batch] prediction error magnitude
        """
        if previous_context is None:
            return torch.zeros(byte_sequence.shape[0], device=byte_sequence.device)
        
        # Predict next semantic from context
        context_seq = previous_context.unsqueeze(1)  # [batch, 1, d_semantic]
        predicted, _ = self.predictor(context_seq)
        predicted_semantic = self.prediction_head(predicted.squeeze(1))
        
        # Get actual semantic
        actual_semantic, _ = self.forward(byte_sequence)
        
        # Surprise = prediction error magnitude (L2 distance)
        surprise = torch.norm(predicted_semantic - actual_semantic, dim=-1)
        
        return surprise


class BrocaProduction(nn.Module):
    """
    Broca's Area - Language production pathway.
    Semantic concepts → Character sequences
    
    Implements autoregressive generation from semantic representations.
    """
    
    def __init__(self, d_semantic: int = 512, d_hidden: int = 512):
        super().__init__()
        self.d_semantic = d_semantic
        self.d_hidden = d_hidden
        
        # Initialize generator hidden state from semantic concept
        self.semantic_to_hidden = nn.Linear(d_semantic, d_hidden * 4)  # 2 layers × (h + c)
        
        # Character embedding for autoregressive generation
        self.char_embed = nn.Embedding(256, 128)
        
        # Main generator (semantic-conditioned)
        self.generator = nn.LSTM(128 + d_semantic, d_hidden, num_layers=2, batch_first=True)
        
        # Output projection to bytes
        self.to_bytes = nn.Linear(d_hidden, 256)
        
        # Selection gate (Basal ganglia-like action selection)
        self.selection_gate = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.Sigmoid()
        )
        
    def forward(self, semantic_concept: torch.Tensor,
                target_bytes: Optional[torch.Tensor] = None,
                max_length: int = 256) -> torch.Tensor:
        """
        Args:
            semantic_concept: [batch, d_semantic] what to express
            target_bytes: [batch, seq_len] for training (teacher forcing)
            max_length: Maximum generation length
        
        Returns:
            logits: [batch, seq_len, 256] if training
            generated: [batch, seq_len] if inference
        """
        batch_size = semantic_concept.shape[0]
        device = semantic_concept.device
        
        # Initialize hidden state from semantic concept
        init = self.semantic_to_hidden(semantic_concept)  # [batch, d_hidden * 4]
        init = init.view(batch_size, 2, 2, self.d_hidden)  # [batch, layers, h/c, hidden]
        h0 = init[:, :, 0, :].permute(1, 0, 2).contiguous()  # [layers, batch, hidden]
        c0 = init[:, :, 1, :].permute(1, 0, 2).contiguous()
        
        if target_bytes is not None:
            # Training mode: teacher forcing
            seq_len = target_bytes.shape[1]
            
            # Embed target bytes
            char_embed = self.char_embed(target_bytes)  # [batch, seq_len, 128]
            
            # Concatenate with semantic context (broadcast)
            semantic_expanded = semantic_concept.unsqueeze(1).expand(-1, seq_len, -1)
            inputs = torch.cat([char_embed, semantic_expanded], dim=-1)
            
            # Generate
            output, _ = self.generator(inputs, (h0, c0))  # [batch, seq_len, d_hidden]
            
            # Apply selection gate (winner-take-all selection)
            gate = self.selection_gate(output)
            output = output * gate
            
            # Project to byte logits
            logits = self.to_bytes(output)  # [batch, seq_len, 256]
            
            return logits
        
        else:
            # Inference mode: autoregressive generation
            generated = []
            h, c = h0, c0
            
            # Start with space byte (32)
            current_byte = torch.full((batch_size,), 32, dtype=torch.long, device=device)
            
            for step in range(max_length):
                # Embed current byte
                char_embed = self.char_embed(current_byte).unsqueeze(1)  # [batch, 1, 128]
                
                # Add semantic context
                semantic_step = semantic_concept.unsqueeze(1)  # [batch, 1, d_semantic]
                input_step = torch.cat([char_embed, semantic_step], dim=-1)
                
                # LSTM step
                output, (h, c) = self.generator(input_step, (h, c))
                
                # Selection gate
                gate = self.selection_gate(output.squeeze(1))
                output_gated = output.squeeze(1) * gate
                
                # Get next byte
                logits = self.to_bytes(output_gated)
                
                # Sample or argmax
                if self.training:
                    probs = F.softmax(logits, dim=-1)
                    current_byte = torch.multinomial(probs, 1).squeeze(-1)
                else:
                    current_byte = torch.argmax(logits, dim=-1)
                
                generated.append(current_byte)
                
                # Stop conditions (newline or period followed by space)
                if step > 5:
                    # Check for sentence end
                    if (current_byte == ord('\n')).all():
                        break
                    if (current_byte == 0).all():  # Padding
                        break
            
            return torch.stack(generated, dim=1)  # [batch, generated_len]


class HippocampalSemanticMemory(nn.Module):
    """
    Hippocampus - Episodic semantic memory storage.
    Stores and retrieves semantic experiences.
    """
    
    def __init__(self, d_semantic: int = 512, capacity: int = 10000):
        super().__init__()
        self.d_semantic = d_semantic
        self.capacity = capacity
        
        # Memory buffer (non-trainable storage)
        self.register_buffer('memory_keys', torch.zeros(capacity, d_semantic))
        self.register_buffer('memory_values', torch.zeros(capacity, d_semantic))
        self.register_buffer('write_ptr', torch.tensor(0, dtype=torch.long))
        self.register_buffer('memory_size', torch.tensor(0, dtype=torch.long))
        
        # Key/value projections
        self.key_proj = nn.Linear(d_semantic, d_semantic)
        self.value_proj = nn.Linear(d_semantic, d_semantic)
        self.query_proj = nn.Linear(d_semantic, d_semantic)
        
    def encode(self, semantic: torch.Tensor, modality: str = 'language'):
        """Store semantic representation in memory."""
        batch_size = semantic.shape[0]
        
        with torch.no_grad():
            for i in range(batch_size):
                key = self.key_proj(semantic[i])
                value = self.value_proj(semantic[i])
                
                # Write to buffer
                ptr = self.write_ptr.item()
                self.memory_keys[ptr] = key
                self.memory_values[ptr] = value
                
                # Update pointer (circular buffer)
                self.write_ptr = (self.write_ptr + 1) % self.capacity
                self.memory_size = min(self.memory_size + 1, self.capacity)
    
    def retrieve(self, query: torch.Tensor, k: int = 5) -> torch.Tensor:
        """Retrieve top-k similar memories."""
        if self.memory_size == 0:
            return query.unsqueeze(1).expand(-1, k, -1)
        
        batch_size = query.shape[0]
        q = self.query_proj(query)  # [batch, d_semantic]
        
        # Compare with stored keys
        valid_size = self.memory_size.item()
        keys = self.memory_keys[:valid_size]  # [valid_size, d_semantic]
        
        # Cosine similarity
        q_norm = F.normalize(q, dim=-1)
        k_norm = F.normalize(keys, dim=-1)
        
        similarities = torch.matmul(q_norm, k_norm.T)  # [batch, valid_size]
        
        # Get top-k
        k_actual = min(k, valid_size)
        top_k_sim, top_k_idx = torch.topk(similarities, k_actual, dim=-1)
        
        # Retrieve values
        retrieved = self.memory_values[top_k_idx]  # [batch, k, d_semantic]
        
        # Pad if needed
        if k_actual < k:
            padding = query.unsqueeze(1).expand(-1, k - k_actual, -1)
            retrieved = torch.cat([retrieved, padding], dim=1)
        
        return retrieved


class BrainLanguageSystem(nn.Module):
    """
    Complete brain-grounded language system.
    
    Integrates:
    - Character patterns → Semantics (Wernicke)
    - Semantic associations (Cortex)
    - Memory storage/retrieval (Hippocampus)
    - Semantics → Text (Broca)
    """
    
    def __init__(self, config: dict):
        super().__init__()
        
        d_pattern = config.get('d_pattern', 256)
        d_semantic = config.get('d_semantic', 512)
        d_hidden = config.get('d_hidden', 512)
        num_concepts = config.get('num_concepts', 1000)
        memory_capacity = config.get('memory_capacity', 10000)
        
        # Comprehension pathway (Wernicke-like)
        self.wernicke = WernickeComprehension(d_pattern, d_semantic, num_concepts)
        
        # Production pathway (Broca-like)
        self.broca = BrocaProduction(d_semantic, d_hidden)
        
        # Semantic memory (Hippocampus)
        self.hippocampus = HippocampalSemanticMemory(d_semantic, memory_capacity)
        
        self.d_semantic = d_semantic
        
    def comprehend(self, text_bytes: torch.Tensor,
                   store_in_memory: bool = True,
                   abstraction_level: Optional[float] = None):
        """
        Understand text (Wernicke pathway).
        text_bytes → patterns → semantics → associations
        """
        # Process through Wernicke
        semantic, concept_activation = self.wernicke(text_bytes, abstraction_level)
        
        # Store in hippocampus (episodic semantic memory)
        if store_in_memory:
            self.hippocampus.encode(semantic, modality='language')
        
        # Update Hebbian associations
        if self.training:
            self.wernicke.associative_network.hebbian_update(concept_activation)
        
        return semantic, concept_activation
    
    def produce(self, semantic_concept: torch.Tensor,
                target_bytes: Optional[torch.Tensor] = None,
                max_length: int = 256) -> torch.Tensor:
        """
        Generate text (Broca pathway).
        Semantic concept → character sequence
        """
        return self.broca(semantic_concept, target_bytes, max_length)
    
    def understand_and_respond(self, input_bytes: torch.Tensor,
                               response_intent: Optional[torch.Tensor] = None,
                               max_length: int = 256) -> torch.Tensor:
        """
        Full comprehension → reasoning → production loop.
        """
        # 1. COMPREHEND (Wernicke)
        input_semantic, input_concepts = self.comprehend(input_bytes)
        
        # 2. REASON (integrate with memory)
        if response_intent is None:
            # Retrieve related concepts from memory
            related_memories = self.hippocampus.retrieve(input_semantic, k=5)
            
            # Integrate input with memories (average)
            response_semantic = (input_semantic + related_memories.mean(dim=1)) / 2
        else:
            response_semantic = response_intent
        
        # 3. PRODUCE (Broca)
        output_bytes = self.produce(response_semantic, max_length=max_length)
        
        return output_bytes
    
    def test_n400(self, context_bytes: torch.Tensor,
                  test_bytes: torch.Tensor) -> torch.Tensor:
        """
        Test N400-like surprise detection.
        
        Args:
            context_bytes: "The cat sat on the..." [batch, seq_len]
            test_bytes: "mat" or "banana" [batch, seq_len]
        
        Returns:
            surprise: [batch] - higher for unexpected continuations
        """
        # Get context semantic
        context_semantic, _ = self.comprehend(context_bytes, store_in_memory=False)
        
        # Compute surprise for test
        surprise = self.wernicke.compute_surprise(test_bytes, context_semantic)
        
        return surprise
    
    def set_abstraction_level(self, level: float):
        """Control abstraction level (the lATL "dimmer")."""
        self.wernicke.abstraction.set_abstraction(level)
    
    def get_abstraction_level(self) -> float:
        """Get current abstraction level."""
        return torch.sigmoid(self.wernicke.abstraction.abstraction_control).item()


def text_to_bytes(text: str, max_len: int = 256) -> torch.Tensor:
    """Convert text string to byte tensor."""
    bytes_list = [ord(c) for c in text[:max_len]]
    if len(bytes_list) < max_len:
        bytes_list += [0] * (max_len - len(bytes_list))
    return torch.tensor(bytes_list, dtype=torch.long)


def bytes_to_text(byte_tensor: torch.Tensor) -> str:
    """Convert byte tensor back to text."""
    if byte_tensor.dim() > 1:
        byte_tensor = byte_tensor[0]  # Take first in batch
    
    chars = []
    for b in byte_tensor.cpu().tolist():
        if b == 0:
            break  # Padding
        if 32 <= b < 127:
            chars.append(chr(b))
        else:
            chars.append('?')
    return ''.join(chars)


def create_brain_language_config() -> dict:
    """Create default configuration for BrainLanguageSystem."""
    return {
        'd_pattern': 256,
        'd_semantic': 512,
        'd_hidden': 512,
        'num_concepts': 1000,
        'memory_capacity': 10000,
    }
