import json
import logging
from datetime import datetime, timezone


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    **fields,
) -> None:
    """Emit a structured event without prompt or provider payload content."""
    logger.log(level, event, extra={"event": event, "event_data": fields})


class StructuredJsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": datetime.fromtimestamp(record.created, timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "event": getattr(record, "event", record.getMessage()),
        }
        payload.update(getattr(record, "event_data", {}))
        return json.dumps(payload, default=str, separators=(",", ":"))


def configure_structured_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(StructuredJsonFormatter())
    logging.basicConfig(level=level, handlers=[handler], force=True)
