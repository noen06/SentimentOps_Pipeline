"""
Lightweight result type shared by the inference pipeline and the API.

Kept free of torch / tensorflow imports so the API layer (and its tests)
can import it without loading heavy ML libraries.
"""

from dataclasses import dataclass, field


@dataclass
class SentimentResult:
    label: str                         # "Negative" | "Neutral" | "Positive"
    confidence: float                  # probability of the predicted class
    probabilities: dict[str, float]    # full distribution over the 3 classes
    predicted_topic: str               # topic predicted by the auxiliary classifier
    cleaned_text: str                  # text after preprocessing
    warnings: list[str] = field(default_factory=list)
