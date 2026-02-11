## Failure Analysis
The system achieved an Avg Groundedness of 0.76. Key failure modes identified during evaluation include:

Knowledge Gap Hallucination: When a ticket asked about topics not covered in the provided .md files (e.g., "dark mode" or "Azure AD SCIM"), the model occasionally hallucinated a standard industry answer instead of strictly adhering to the "I don't know" instruction.

Context Mixing: In cases with overlapping technical terms (like "API Token" vs "Webhook Secret"), the retriever occasionally pulled chunks from both. The LLM then blended these distinct processes into a single, technically incorrect set of instructions.

Irrelevant Retrieval (Noise): For very short ticket bodies, the vector search sometimes retrieved documents with high keyword overlap but low semantic relevance (e.g., retrieving "Billing" docs for a "Refund" request when the refund policy wasn't explicitly in the KB).

P0 Tone Over-Correction: In P0 adversarial cases, the instruction to be "extremely concise" occasionally led the model to omit required troubleshooting steps found in the context, lowering the relevance score.