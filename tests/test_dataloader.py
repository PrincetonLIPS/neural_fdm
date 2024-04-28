import pytest
from your_module import JsonDataset


@pytest.fixture
def json_dataset():
    # Create a sample JSON dataset for testing
    dataset = JsonDataset("/path/to/dataset.json")
    return dataset


def test_json_dataset_load(json_dataset):
    # Test if the dataset is loaded successfully
    assert len(json_dataset) > 0


def test_json_dataset_get_item(json_dataset):
    # Test if the __getitem__ method returns the correct item
    item = json_dataset[0]
    assert isinstance(item, dict)


def test_json_dataset_len(json_dataset):
    # Test if the __len__ method returns the correct length
    assert len(json_dataset) == json_dataset.get_length()


def test_train_test_split(json_dataset):
    # Test if the train-test split method works correctly
    train_dataset, test_dataset = json_dataset.train_test_split(0.8)
    assert len(train_dataset) + len(test_dataset) == len(json_dataset)
