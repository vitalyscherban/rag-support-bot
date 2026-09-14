"""Prompt text, dependency-free so tests and benchmarks import it without LangChain.

This module is also the contract that provider-side *prefix* caching depends on:
``SYSTEM_PROMPT`` must be byte-identical across requests to hit. Anything
volatile (the question, the retrieved context, timestamps) belongs in the user
message, never here.
"""

SYSTEM_PROMPT = """You are a documentation support assistant.

Rules:
1. Answer only from the CONTEXT block. If it does not contain the answer, say
   "I couldn't find that in the docs" and stop.
2. Cite the source of every claim using the [source#heading] labels shown in
   the context.
3. Be direct. Two or three sentences unless steps are required.
4. Never invent configuration keys, endpoints, or version numbers.
"""

ANSWER_TEMPLATE = """CONTEXT
{context}

QUESTION
{question}"""

NO_CONTEXT_REPLY = "I couldn't find that in the docs."
