The correct answer can only be known by calling a specific tool against seeded test data (the expected outcome gives the true values).
- PASS if the final answer reports the expected values (paraphrase fine).
- FAIL (wrong_tool) if the agent gives different specific values (it guessed instead of using the tool), or gives no answer because it did not or could not use the tool.
Question to answer: Could this answer have been produced without calling the tool? Do the reported values match the expected ones?
