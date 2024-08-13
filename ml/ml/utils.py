from json import load


def load_configuration() -> dict:
    """
    Loads the configuration file and returns it as a dictionary.
    :return:
    """
    with open("configuration.json") as file:
        return load(file)
