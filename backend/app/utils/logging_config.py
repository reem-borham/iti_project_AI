import logging
import sys

def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Configure and return the root logger for the application."""
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True,
    )
    
    logger = logging.getLogger("rag_backend")
    logger.setLevel(numeric_level)
    return logger
