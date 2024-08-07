import logging
import coloredlogs


def configure_logging(level=logging.INFO):
    log_format = '[ %(asctime)s - %(levelname)s ] %(message)s'

    logging.basicConfig(
        level=level,
        format=log_format,
        datefmt='%H:%M:%S.%f',
        handlers=[
            logging.StreamHandler()
        ]
    )

    field_styles = {
        'asctime': {'color': 'green'},
        'hostname': {'color': 'magenta'},
        'levelname': {'color': 'white', 'bold': True},
        'name': {'color': 'blue'},
        'programname': {'color': 'cyan'},
        'username': {'color': 'yellow'}
    }
    level_styles = {
        'debug': {'color': 'blue'},
        'info': {'color': 'white'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red', 'bold': True},
        'critical': {'color': 'black', 'bold': True, 'background': 'red'}
    }

    coloredlogs.install(level=level, fmt=log_format, datefmt='%H:%M:%S.%f', field_styles=field_styles, level_styles=level_styles)


configure_logging()

logger = logging.getLogger(__name__)
