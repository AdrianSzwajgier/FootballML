from unittest.mock import patch

import pandas as pd
import pytest

from ml.ml.main import load_data


@patch('pandas.read_csv')
def test_load_data(mock_read_csv):
    df = pd.DataFrame({"col1": [1, 2, 3]})
    mock_read_csv.return_value = df

    result = load_data("dummy_path.csv")

    mock_read_csv.assert_called_with("dummy_path.csv")
    assert result.equals(df)

    mock_read_csv.side_effect = FileNotFoundError
    with pytest.raises(FileNotFoundError):
        load_data('non_existent_file.csv')
