__version__ = "0.2.0"


def is_compatible_schema_version(version: str | None) -> bool:
    """True if a dataset written under `version` has this schema's major.minor version."""
    return version is not None and version.split(".")[:2] == __version__.split(".")[:2]
