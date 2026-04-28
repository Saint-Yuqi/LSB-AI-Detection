"""
Review engine and AI verifier subproject.

Three-layer architecture:
  - Business JSONL (verifier_examples_{family}.jsonl)
  - Render asset layer (shared context, crop cache, asset manifest)
  - Model-input layer (verifier_chat_{family}.jsonl)

Three isolated verifier families:
  - satellite_mv  (candidate-level mask verification)
  - satellite_ev  (image-level exhaustivity verification)
  - stream_ev     (image-level stream exhaustivity verification)
"""
