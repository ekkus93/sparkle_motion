"""sparkle_motion package

Small package entry for the Sparkle Motion project. Keeps package namespace
and exposes version.
"""
__all__ = ["schemas", "schema_registry", "prompt_templates"]
__version__ = "0.0.0"

from . import schemas
from . import schema_registry
from . import prompt_templates
