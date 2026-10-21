The user asked something whose answer IS in the ground truth.
- PASS if the final answer states the expected fact correctly (paraphrase is fine) and asserts nothing that contradicts the ground truth.
- FAIL (hallucination) if the expected fact is missing, wrong, or contradicted, or if the agent adds specific claims the ground truth does not support.
Question to answer: Does the response state the expected fact? Does it assert anything not supported by the ground truth? Quote it.
