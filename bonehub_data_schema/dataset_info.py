class DatasetInfo(dict):
    """Data dict for a dataset."""

    valid_keys = {
        "dataset_id",
        "name",
        "description",
        "license",
        "url",
        "version",
        "release_date",
        "paper",
    }

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

        if not set(kwargs.keys()).issubset(self.valid_keys):
            raise ValueError(f"Invalid keys: {set(kwargs.keys()) - self.valid_keys}. Valid keys are: {self.valid_keys}")

        if (dataset_id := kwargs.get("dataset_id")) is not None:
            self["dataset_id"] = dataset_id

        if (name := kwargs.get("name")) is not None:
            self["name"] = name
        if (description := kwargs.get("description")) is not None:
            self["description"] = description
        if (license := kwargs.get("license")) is not None:
            self["license"] = license
        if (url := kwargs.get("url")) is not None:
            self["url"] = url
        if (version := kwargs.get("version")) is not None:
            self["version"] = version
        if (release_date := kwargs.get("release_date")) is not None:
            self["release_date"] = release_date
        if (paper := kwargs.get("paper")) is not None:
            self["paper"] = paper

    def __setitem__(self, key, value):
        if key not in self.valid_keys:
            raise KeyError(f"Invalid key: {key}. Valid keys are: {self.valid_keys}")
        super().__setitem__(key, value)
