

from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class Document:
    text: str
    metadata: Dict[str, Optional[str]]