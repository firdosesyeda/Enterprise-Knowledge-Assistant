"""
LLM Service
Handles answer generation using Ollama (local Llama 3).

Design choices:
- Llama 3 via Ollama: fully local, free, private, no API key needed
- Ollama exposes an OpenAI-compatible API, so we use the OpenAI client with a custom base_url
- Structured prompt with clear instructions prevents hallucination
- System prompt establishes grounding rules: answer ONLY from provided context
- Confidence estimation based on retrieval score + answer analysis
- Conversation memory via message history
"""
import logging
from typing import List, Dict, Any, Optional, Tuple

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are an Enterprise Knowledge Assistant. Your job is to answer employee questions accurately based ONLY on the provided document excerpts.

RULES:
1. Answer ONLY using the information in the provided context excerpts
2. If the context does not contain enough information to answer, say: "I couldn't find information about this in the available documents."
3. Never make up facts, policies, or numbers
4. Be concise and direct
5. If the answer spans multiple documents, synthesize them clearly
6. Always ground your answer in the provided excerpts
7. If you are uncertain, say so explicitly

FORMAT:
- Give a clear, direct answer
- Use bullet points for multi-part answers
- Reference which document the information comes from when relevant"""


class LLMService:
    def __init__(self):
        self.client = OpenAI(
            api_key="ollama",  # Ollama doesn't need a real key, but the client requires one
            base_url=settings.ollama_base_url,
        )
        self.model_name = settings.llm_model
        self.conversation_histories: Dict[str, List[Dict]] = {}

    def build_context(self, chunks: List[Dict[str, Any]]) -> str:
        """Format retrieved chunks into a structured context block."""
        if not chunks:
            return "No relevant documents found."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            doc_name = chunk.get("doc_name", "Unknown")
            doc_type = chunk.get("doc_type", "Document")
            page = chunk.get("page_num")
            page_str = f", Page {page}" if page else ""
            score = chunk.get("final_score", chunk.get("semantic_score", 0))

            context_parts.append(
                f"[Excerpt {i} | Source: {doc_name} ({doc_type}){page_str} | Relevance: {score:.2f}]\n"
                f"{chunk['text']}\n"
                f"{'─' * 60}"
            )

        return "\n".join(context_parts)

    def generate_answer(
        self,
        question: str,
        chunks: List[Dict[str, Any]],
        conversation_id: Optional[str] = None,
    ) -> Tuple[str, float]:
        """
        Generate an answer from retrieved chunks.
        Returns (answer, confidence_score).
        """
        context = self.build_context(chunks)

        user_message = (
            f"Context from enterprise documents:\n\n{context}\n\n"
            f"{'═' * 60}\n\n"
            f"Employee Question: {question}\n\n"
            f"Please answer the question based strictly on the context above."
        )

        # Build OpenAI-compatible chat history
        history = self._get_history(conversation_id)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        messages.extend(history)
        messages.append({"role": "user", "content": user_message})

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2,
                max_tokens=1024,
            )
            answer = response.choices[0].message.content

            # Persist conversation history
            if conversation_id:
                updated = list(history) + [
                    {"role": "user", "content": user_message},
                    {"role": "assistant", "content": answer},
                ]
                # Keep last 20 turns (10 exchanges) to avoid context overflow
                self.conversation_histories[conversation_id] = updated[-20:]

            confidence = self._estimate_confidence(answer, chunks)
            return answer, confidence

        except Exception as e:
            err = str(e).lower()
            if "connection" in err or "refused" in err or "connect" in err:
                raise ValueError(
                    "Cannot connect to Ollama. Make sure Ollama is running.\n"
                    "Start it with: ollama serve"
                )
            logger.error(f"LLM error: {e}")
            raise

    def _get_history(self, conversation_id: Optional[str]) -> List[Dict]:
        """Return OpenAI-formatted conversation history."""
        if not conversation_id:
            return []
        return list(self.conversation_histories.get(conversation_id, []))

    def _estimate_confidence(self, answer: str, chunks: List[Dict]) -> float:
        """
        Heuristic confidence score [0.0, 1.0].

        Factors:
        - Top retrieval score
        - Whether the model indicated uncertainty
        - Number of relevant chunks found
        """
        if not chunks:
            return 0.0

        # Low confidence phrases
        low_confidence_phrases = [
            "couldn't find", "not find", "no information", "not available",
            "don't have", "unable to", "not mentioned", "not specified",
            "unclear", "uncertain",
        ]

        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in low_confidence_phrases):
            return 0.15

        # Base from retrieval quality
        top_score = max(
            c.get("final_score", c.get("semantic_score", 0)) for c in chunks
        )

        # Bonus for multiple strong sources
        strong_chunks = sum(
            1 for c in chunks
            if c.get("final_score", c.get("semantic_score", 0)) > 0.5
        )

        confidence = min(
            top_score * 0.7 + (strong_chunks / max(len(chunks), 1)) * 0.3,
            0.99,
        )

        return round(confidence, 3)

    def rewrite_query(self, question: str) -> str:
        """
        Query rewriting: expand abbreviations, make question more precise.
        This improves retrieval for ambiguous or short questions.
        """
        prompt = (
            f"Rewrite this question to be more specific and search-friendly for "
            f"an enterprise document search system. Keep it concise (1-2 sentences). "
            f"Return only the rewritten question, nothing else.\n\nQuestion: {question}"
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                max_tokens=128,
            )
            rewritten = response.choices[0].message.content.strip()
            logger.debug(f"Query rewritten: '{question}' -> '{rewritten}'")
            return rewritten
        except Exception:
            return question  # Fall back to original
