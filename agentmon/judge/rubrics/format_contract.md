The caller depends on a specific output shape (deterministic schema checks have already passed).
- PASS if the content inside the required format is also correct per the expected outcome.
- FAIL (hallucination) if the structured content is factually wrong per the ground truth.
