__version__ = "0.2.0"


def is_compatible_schema_version(version: str | None) -> bool:
    """Whether a dataset written under `version` can be read by this schema.

    Label values and status codes are plain integers, so a dataset written under another
    scheme would not fail validation - it would silently be read with the wrong meaning.
    Versions below 1.0 may change either within a minor release, so the major and minor
    numbers must match; a dataset without a recorded version predates versioning.
    """
    return version is not None and version.split(".")[:2] == __version__.split(".")[:2]
