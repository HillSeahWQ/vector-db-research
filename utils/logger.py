import logging
import os
import sys
from logging.handlers import RotatingFileHandler

# Create a logs directory if it doesn't exist
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'logs')
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, 'app.log')


class SafeStreamHandler(logging.StreamHandler):
    """A StreamHandler that replaces unencodable characters instead of crashing."""
    def emit(self, record):
        try:
            msg = self.format(record)
            stream = self.stream
            # Write with replacement for bad characters
            stream.write(msg.encode(stream.encoding, errors="replace").decode(stream.encoding, errors="replace"))
            stream.write(self.terminator)
            self.flush()
        except Exception:
            self.handleError(record)


def get_logger(name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:  # Prevent adding multiple handlers in case of multiple imports
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

        # Console handler (safe Unicode handling)
        console_handler = SafeStreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File handler with rotation (always UTF-8 safe)
        file_handler = RotatingFileHandler(
            LOG_FILE, maxBytes=5*1024*1024, backupCount=3, encoding="utf-8"
        )
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
