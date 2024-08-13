from unittest.mock import patch

from ml.ml.utils import load_configuration


@patch("ml.ml.utils.open")
@patch("ml.ml.utils.load")
def test_load_configuration(mock_load, mock_open):
    load_configuration()
    mock_open.assert_called_once_with("configuration.json")
    mock_load.assert_called_once_with(mock_open().__enter__())
