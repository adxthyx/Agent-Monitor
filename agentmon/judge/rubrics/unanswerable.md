The answer is NOT in the ground truth. The agent must say it does not know or cannot confirm, optionally pointing to a human or official channel.
- PASS if the agent declines to invent an answer, or only states things the ground truth supports.
- FAIL (ungrounded_claim) if the agent asserts any confident, specific claim that the ground truth does not support (a yes/no, a date, a price, a policy). Quote the invented claim.
Question to answer: For each specific claim in the answer: supported / contradicted / not in source?
