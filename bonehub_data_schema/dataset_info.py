class DatasetInfo(dict):
    """Data dict for a dataset."""

    valid_keys = (
        "dataset_id",
        "name",
        "description",
        "license",
        "url",
        "version",
        "release_date",
        "paper",
    )

    def __init__(
        self,
        **kwargs,
    ):
        """
        Initialize a DatasetInfo dict with required information and optional information.
        Args:
            **kwargs: Optional keyword arguments for additional dataset information. Valid keys are:
                - dataset_id (int): Unique identifier for the dataset in BoneHub.
                - name (str): Name of the dataset
                - description (str): Description of the dataset
                - license (str): License of the dataset
                - url (str): URL of the dataset
                - version (str): Version of the dataset
                - release_date (str): Release date of the dataset
                - paper (str): Paper reference for the dataset
        """
        super().__init__()

        if not set(kwargs.keys()).issubset(set(self.valid_keys)):
            raise ValueError(f"Invalid keys: {set(kwargs.keys()) - set(self.valid_keys)}. Valid keys are: {self.valid_keys}")

        if dataset_id := kwargs.get("dataset_id"):
            self["dataset_id"] = dataset_id
        if name := kwargs.get("name"):
            self["name"] = name
        if description := kwargs.get("description"):
            self["description"] = description
        if license := kwargs.get("license"):
            self["license"] = license
        if url := kwargs.get("url"):
            self["url"] = url
        if version := kwargs.get("version"):
            self["version"] = version
        if release_date := kwargs.get("release_date"):
            self["release_date"] = release_date
        if paper := kwargs.get("paper"):
            self["paper"] = paper

    def __setitem__(self, key, value):
        if key not in self.valid_keys:
            raise KeyError(f"Invalid key: {key}. Valid keys are: {self.valid_keys}")
        if key == "dataset_id" and not isinstance(value, int):
            raise ValueError(f"dataset_id must be an integer. Got {type(value)} instead.")
        super().__setitem__(key, value)

    def sorted(self) -> dict:
        """Return the keys in the DatasetInfo dict sorted according to valid_keys order."""
        sorted_dict = {key: self[key] for key in self.valid_keys if key in self}
        return sorted_dict
