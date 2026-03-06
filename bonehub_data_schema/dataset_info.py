from pydantic import BaseModel, Field, ConfigDict


class DatasetInfo(BaseModel):
    """Data model for a dataset with automatic validation."""

    dataset_id: int | None = Field(None, description="Unique identifier for the dataset in BoneHub")
    name: str | None = Field(None, description="Name of the dataset")
    description: str | None = Field(None, description="Description of the dataset")
    url: str | None = Field(None, description="URL to access the dataset")
    paper: str | None = Field(None, description="URL to the paper describing the dataset")
    country: str | None = Field(None, description="Country of origin")
    release_date: str | None = Field(None, description="Release date of the dataset")
    version: str | None = Field(None, description="Version of the dataset")
    remarks: str | None = Field(None, description="Remarks about the dataset")
    modality: str | None = Field(None, description="Imaging modality used (e.g., CT, MRI)")
    license: str | None = Field(None, description="License of the dataset")

    model_config = ConfigDict(validate_assignment=True, strict=True)

    def sorted_dict(self) -> dict:
        """Return the dataset info sorted by field definition order, excluding None values."""
        return {key: getattr(self, key) for key in self.__class__.model_fields if getattr(self, key) is not None}
