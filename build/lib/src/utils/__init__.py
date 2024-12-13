import time
from ..utils.config_parser import Config, ConfigReader


def measure_time(start_time=None):
    if start_time is None:
        return time.time()

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)

    return f"{int(hours):02d}:{int(minutes):02d}:{int(seconds):02d}"


__all__ = ["measure_time", "Config", "ConfigReader"]
