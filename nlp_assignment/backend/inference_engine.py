"""
Project Sentinel Inference Engine
Mechanistic Interpretability for Real-Time Hallucination Detection

This module implements:
1. Forward-hook listener on every transformer layer
2. Mahalanobis distance computation with Ledoit-Wolf shrinkage
3. Logit-Lens entropy tracking
4. Dynamic hallucination flagging (σ > 3.0)
5. PCA-based causal intervention steering
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
import logging

logger = logging.getLogger(__name__)


@dataclass
class DriftAnalysis:
    """Container for per-token drift analysis results."""
    layer_id: int
    token_id: int
    token_text: str
    drift_score: float
    is_hallucination: bool
    entropy: float
    logit_lens_entropy: List[float]  # Per-layer entropy
    hidden_state: torch.Tensor  # h_{l,t}


@dataclass
class InterventionResult:
    """Container for causal intervention results."""
    baseline_logits: torch.Tensor
    steered_logits: torch.Tensor
    baseline_probs: torch.Tensor
    steered_probs: torch.Tensor
    steering_magnitude: float
    semantic_shift: float


class FaithfulManifold:
    """
    Pre-computed manifold of faithful (non-hallucinated) activations.
    Uses Ledoit-Wolf shrinkage for robust covariance estimation.
    """

    def __init__(self, hidden_dim: int, n_layers: int):
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.means: Dict[int, np.ndarray] = {}
        self.covs: Dict[int, np.ndarray] = {}
        self.pca_models: Dict[int, PCA] = {}
        self.is_fitted = False

    def fit(self, activations: Dict[int, np.ndarray]):
        """
        Fit the manifold from faithful activations.
        
        Args:
            activations: Dict mapping layer_id -> (N, hidden_dim) array of faithful activations
        """
        for layer_id, acts in activations.items():
            # Compute mean
            self.means[layer_id] = np.mean(acts, axis=0)
            
            # Ledoit-Wolf shrinkage for robust covariance
            lw = LedoitWolf()
            cov, _ = lw.fit(acts).covariance_, lw.shrinkage_
            self.covs[layer_id] = cov
            
            # Fit PCA for steering
            pca = PCA(n_components=min(acts.shape[0] // 2, self.hidden_dim))
            pca.fit(acts - self.means[layer_id])
            self.pca_models[layer_id] = pca
        
        self.is_fitted = True
        logger.info(f"Faithful manifold fitted for {len(activations)} layers")

    def mahalanobis_distance(self, layer_id: int, activation: np.ndarray) -> float:
        """
        Compute Mahalanobis distance from faithful manifold.
        
        Args:
            layer_id: Layer identifier
            activation: (hidden_dim,) activation vector
            
        Returns:
            Mahalanobis distance (scalar)
        """
        if not self.is_fitted or layer_id not in self.means:
            return 0.0
        
        mean = self.means[layer_id]
        cov = self.covs[layer_id]
        
        # Mahalanobis distance: sqrt((x - μ)^T Σ^{-1} (x - μ))
        diff = activation - mean
        try:
            cov_inv = np.linalg.inv(cov + 1e-6 * np.eye(cov.shape[0]))
            distance = np.sqrt(np.dot(diff, np.dot(cov_inv, diff)))
        except np.linalg.LinAlgError:
            distance = np.linalg.norm(diff)
        
        return float(distance)

    def project_to_faithful_subspace(self, layer_id: int, activation: np.ndarray) -> np.ndarray:
        """
        Project activation onto the faithful subspace using PCA.
        
        Args:
            layer_id: Layer identifier
            activation: (hidden_dim,) activation vector
            
        Returns:
            Projected activation in faithful subspace
        """
        if not self.is_fitted or layer_id not in self.pca_models:
            return activation
        
        mean = self.means[layer_id]
        pca = self.pca_models[layer_id]
        
        # Project to PCA space and back
        centered = activation - mean
        pca_coords = pca.transform(centered.reshape(1, -1))[0]
        projected = pca.inverse_transform(pca_coords.reshape(1, -1))[0]
        
        return projected + mean


class InferenceEngine:
    """
    Real-time inference engine with mechanistic interpretability hooks.
    """

    def __init__(self, model_name: str = "facebook/opt-1.3b", device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        self.faithful_manifold = None
        
        # Hook storage
        self.layer_activations: Dict[int, torch.Tensor] = {}
        self.hook_handles: List = []
        
        # Configuration
        self.drift_threshold = 3.0  # σ > 3.0 for hallucination flag
        self.n_layers = 24  # OPT-1.3b has 24 layers
        
        self._load_model()

    def _load_model(self):
        """Load the transformer model and tokenizer."""
        logger.info(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            output_hidden_states=True,
            torch_dtype=torch.float16
        ).to(self.device)
        self.model.eval()
        logger.info("Model loaded successfully")

    def _create_forward_hook(self, layer_id: int):
        """Create a forward hook for a specific layer."""
        def hook(module, input, output):
            # Extract hidden states from output
            if isinstance(output, tuple):
                hidden_state = output[0]
            else:
                hidden_state = output
            
            self.layer_activations[layer_id] = hidden_state.detach().cpu()
        
        return hook

    def register_hooks(self):
        """Register forward hooks on all transformer layers."""
        logger.info("Registering forward hooks on all layers")
        
        # Register hooks on decoder layers
        for layer_id, layer in enumerate(self.model.model.decoder.layers):
            handle = layer.register_forward_hook(self._create_forward_hook(layer_id))
            self.hook_handles.append(handle)
        
        logger.info(f"Registered {len(self.hook_handles)} hooks")

    def unregister_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hook_handles:
            handle.remove()
        self.hook_handles.clear()
        logger.info("Hooks unregistered")

    def initialize_manifold(self, faithful_activations: Dict[int, np.ndarray]):
        """
        Initialize the faithful manifold from pre-computed activations.
        
        Args:
            faithful_activations: Dict mapping layer_id -> (N, hidden_dim) array
        """
        self.faithful_manifold = FaithfulManifold(
            hidden_dim=self.model.config.hidden_size,
            n_layers=self.n_layers
        )
        self.faithful_manifold.fit(faithful_activations)

    def compute_logit_lens_entropy(self, hidden_state: torch.Tensor) -> List[float]:
        """
        Compute entropy of logits at each layer (Logit-Lens).
        
        Args:
            hidden_state: (seq_len, hidden_dim) tensor
            
        Returns:
            List of entropy values per layer
        """
        entropies = []
        
        # Use the model's language modeling head to compute logits
        with torch.no_grad():
            logits = self.model.lm_head(hidden_state)  # (seq_len, vocab_size)
            probs = F.softmax(logits, dim=-1)
            entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=-1)
            entropies = entropy.cpu().numpy().tolist()
        
        return entropies

    def analyze_token(self, token_id: int, token_text: str) -> DriftAnalysis:
        """
        Analyze a single generated token for hallucination signals.
        
        Args:
            token_id: Token ID from tokenizer
            token_text: Decoded token text
            
        Returns:
            DriftAnalysis object with drift scores and flags
        """
        if not self.faithful_manifold:
            raise ValueError("Faithful manifold not initialized")
        
        # Compute drift scores across all layers
        drift_scores = []
        for layer_id in range(self.n_layers):
            if layer_id in self.layer_activations:
                hidden = self.layer_activations[layer_id]
                # Take the last token's activation
                last_activation = hidden[-1, :].cpu().numpy()
                drift = self.faithful_manifold.mahalanobis_distance(layer_id, last_activation)
                drift_scores.append(drift)
        
        # Compute statistics
        drift_scores = np.array(drift_scores)
        mean_drift = np.mean(drift_scores)
        std_drift = np.std(drift_scores)
        normalized_drift = (mean_drift - np.mean(drift_scores)) / (std_drift + 1e-6)
        
        # Flag hallucination if drift exceeds threshold
        is_hallucination = normalized_drift > self.drift_threshold
        
        # Compute entropy
        if self.n_layers - 1 in self.layer_activations:
            hidden = self.layer_activations[self.n_layers - 1]
            entropies = self.compute_logit_lens_entropy(hidden)
            entropy = entropies[-1] if entropies else 0.0
        else:
            entropy = 0.0
        
        return DriftAnalysis(
            layer_id=self.n_layers - 1,
            token_id=token_id,
            token_text=token_text,
            drift_score=float(mean_drift),
            is_hallucination=is_hallucination,
            entropy=entropy,
            logit_lens_entropy=drift_scores.tolist(),
            hidden_state=self.layer_activations.get(self.n_layers - 1, torch.zeros(1))
        )

    def causal_intervention(self, peak_layer_id: int) -> InterventionResult:
        """
        Perform causal intervention by steering the peak drift layer.
        
        Args:
            peak_layer_id: Layer with highest drift to intervene on
            
        Returns:
            InterventionResult with baseline and steered outputs
        """
        if not self.faithful_manifold or peak_layer_id not in self.layer_activations:
            raise ValueError("Cannot perform intervention without manifold or activations")
        
        # Baseline: original hidden state
        baseline_hidden = self.layer_activations[peak_layer_id].to(self.device)
        
        # Intervention: project to faithful subspace
        baseline_np = baseline_hidden[-1, :].cpu().numpy()
        steered_np = self.faithful_manifold.project_to_faithful_subspace(
            peak_layer_id, baseline_np
        )
        steered_hidden = torch.tensor(steered_np, dtype=baseline_hidden.dtype).to(self.device)
        
        # Compute logits for both
        with torch.no_grad():
            baseline_logits = self.model.lm_head(baseline_hidden[-1, :])
            steered_logits = self.model.lm_head(steered_hidden)
        
        baseline_probs = F.softmax(baseline_logits, dim=-1)
        steered_probs = F.softmax(steered_logits, dim=-1)
        
        # Steering magnitude: L2 distance in hidden space
        steering_magnitude = float(torch.norm(steered_hidden - baseline_hidden[-1, :]))
        
        # Semantic shift: KL divergence between distributions
        semantic_shift = float(F.kl_div(
            torch.log(steered_probs + 1e-10),
            baseline_probs,
            reduction='batchmean'
        ))
        
        return InterventionResult(
            baseline_logits=baseline_logits.cpu(),
            steered_logits=steered_logits.cpu(),
            baseline_probs=baseline_probs.cpu(),
            steered_probs=steered_probs.cpu(),
            steering_magnitude=steering_magnitude,
            semantic_shift=semantic_shift
        )

    def generate_with_analysis(
        self,
        prompt: str,
        max_new_tokens: int = 50,
        temperature: float = 0.7
    ) -> Tuple[str, List[DriftAnalysis]]:
        """
        Generate text with real-time drift analysis.
        
        Args:
            prompt: Input prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            
        Returns:
            (generated_text, list_of_drift_analyses)
        """
        self.register_hooks()
        
        try:
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            analyses = []
            
            with torch.no_grad():
                for _ in range(max_new_tokens):
                    outputs = self.model(**inputs, output_hidden_states=True)
                    next_token_logits = outputs.logits[:, -1, :]
                    next_token_logits = next_token_logits / temperature
                    probs = F.softmax(next_token_logits, dim=-1)
                    next_token_id = torch.multinomial(probs, num_samples=1)[0, 0]
                    
                    # Analyze the generated token
                    token_text = self.tokenizer.decode([next_token_id])
                    analysis = self.analyze_token(next_token_id.item(), token_text)
                    analyses.append(analysis)
                    
                    # Append to input for next iteration
                    inputs = torch.cat([inputs, next_token_id.unsqueeze(0).unsqueeze(0)], dim=1)
            
            generated_text = self.tokenizer.decode(inputs[0], skip_special_tokens=True)
            return generated_text, analyses
        
        finally:
            self.unregister_hooks()


def create_mock_faithful_activations(n_samples: int = 1000, hidden_dim: int = 2048, n_layers: int = 24) -> Dict[int, np.ndarray]:
    """
    Create mock faithful activations for testing (in production, use real faithful data).
    
    Args:
        n_samples: Number of samples
        hidden_dim: Hidden dimension size
        n_layers: Number of layers
        
    Returns:
        Dict mapping layer_id -> (n_samples, hidden_dim) array
    """
    activations = {}
    for layer_id in range(n_layers):
        # Simulate faithful activations with Gaussian distribution
        acts = np.random.randn(n_samples, hidden_dim).astype(np.float32)
        activations[layer_id] = acts
    
    return activations
