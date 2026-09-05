"""AffectLab: a real-time affective computing laboratory for ordinary webcams.

The package turns a video stream into a multimodal picture of a person's
affective state:

* facial **Action Units** (Facial Action Coding System) estimated from
  MediaPipe blendshapes,
* categorical **emotion estimates** from interchangeable backends (an
  explainable FACS rule set, the FER+ convolutional network, DeepFace, or an
  ensemble),
* dimensional **affect** on Russell's circumplex (valence and arousal) with
  temporal smoothing and emotion-dynamics statistics (inertia, variability,
  switch rate),
* contactless **heart rate** by remote photoplethysmography (rPPG),
* **blink** and eyelid-closure metrics (blink rate, PERCLOS) and head pose.

Everything runs locally. No frame ever leaves the machine.
"""

from affectlab.types import (
    EMOTIONS,
    Affect,
    BBox,
    Dynamics,
    EmotionEstimate,
    FaceObservation,
    FrameResult,
    HeadPose,
    Vitals,
)

__version__ = "2.0.0"

__all__ = [
    "EMOTIONS",
    "Affect",
    "BBox",
    "Dynamics",
    "EmotionEstimate",
    "FaceObservation",
    "FrameResult",
    "HeadPose",
    "Vitals",
    "__version__",
]
