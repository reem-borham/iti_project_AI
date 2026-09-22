import logging
from typing import Optional

logger = logging.getLogger("rag_backend.generation")

PROMPT_TEMPLATE = """You are a teaching assistant for Python and Data Science.
Your job is to answer the user's question using ONLY the context passages provided below.
For every fact you state, cite the source document and page number like this: [Source: <filename>, p.<page>].
If the answer cannot be found in the provided context, respond with: "I don't know based on the available textbook content."
Do NOT use any knowledge outside the provided context.

Context:
{context}

Question: {question}

Answer (with citations):"""


class GenerationService:
    """Manages Ollama LLM interactions and grounded response generation."""

    def __init__(self, base_url: str = "http://localhost:11434", model: str = "llama3.2:1b"):
        self.base_url = base_url
        self.model = model
        self.client = None
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize the Ollama client."""
        try:
            import ollama
            self.client = ollama.Client(host=self.base_url)
            logger.info(f"Ollama client initialized with host={self.base_url}, model={self.model}")
        except Exception as exc:
            logger.error(f"Failed to initialize Ollama client: {exc}")
            self.client = None

    def is_connected(self) -> bool:
        """Check if Ollama server is accessible."""
        if not self.client:
            return False
        try:
            self.client.list()
            return True
        except Exception:
            return False

    def build_prompt(self, question: str, context: str) -> str:
        """Combine user question with retrieved context into the grounded prompt template."""
        clean_context = context.strip() if context else "No relevant context found in textbooks."
        return PROMPT_TEMPLATE.format(context=clean_context, question=question.strip())

    def generate_answer(self, question: str, context: str) -> str:
        """
        Generate grounded answer using the Ollama LLM.
        """
        if not self.client:
            return "Error: LLM client is not initialized."

        prompt = self.build_prompt(question=question, context=context)

        try:
            response = self.client.generate(
                model=self.model,
                prompt=prompt,
                options={
                    "temperature": 0.1,
                    "top_p": 0.9,
                },
            )
            # Handle both object-style (ollama >= 0.2) and dict-style responses
            if hasattr(response, "response"):
                return response.response.strip()
            return response.get("response", "").strip()
        except Exception as exc:
            logger.error(f"Ollama generation failed: {exc}")
            return f"Error communicating with Ollama service at {self.base_url}: {exc}"
