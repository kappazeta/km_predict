# vim: set tabstop=8 softtabstop=0 expandtab shiftwidth=4 smarttab

# Logging functionality.
#
# Copyright 2021 KappaZeta Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import sys
import logging
from version import __version__


class LogInfoFilter(logging.Filter):
    """A logging filter that discards warnings and errors."""

    def filter(self, record):
        """
        Discard warnings and errors
        :param record: Log record
        :return: True for records to keep, False for records to drop
        """
        """False on warning or error records."""
        return record.levelno in (logging.DEBUG, logging.INFO)


class Loggable(object):
    """A class that uses the logging functionality."""

    def __init__(self, log_module_abbrev):
        self.log = logging.getLogger(log_module_abbrev)


def init_logging(verbosity, app_name, app_abbrev, logfile=None):
    """
    Initialize logging, based on verbosity level
    :param verbosity: one of the following: logging.ERROR, logging.WARNING, logging.INFO, logging.DEBUG
    :param app_name: Application title
    :param app_abbrev: Application abbreviation to use for hierarchical logging
    :param logfile: Path to a log file (None (disabled) by default)
    :return: Log instance
    """
    """Initialize logging, based on verbosity level."""
    # Configure logging
    if verbosity == 0:
        log_level = logging.WARNING
    elif verbosity == 1:
        log_level = logging.INFO
    elif verbosity > 2:
        log_level = logging.DEBUG
    else:
        log_level = logging.NOTSET

    log = logging.getLogger(app_abbrev)
    log.setLevel(log_level)

    # Create log formatters.
    log_formatter = logging.Formatter('%(asctime)s: %(levelname)s: %(name)s: %(message)s')
    stdout_formatter = logging.Formatter('%(levelname)s: %(name)s: %(message)s')

    # Info, debug messages to stdout (if we have enough verbosity).
    stdout_handler = logging.StreamHandler(sys.stdout)
    stdout_handler.setLevel(log_level)
    stdout_handler.setFormatter(stdout_formatter)
    stdout_handler.addFilter(LogInfoFilter())
    log.addHandler(stdout_handler)

    # Warnings and errors to stderr.
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setLevel(logging.WARNING)
    stderr_handler.setFormatter(stdout_formatter)
    log.addHandler(stderr_handler)

    # Regular log file.
    if logfile:
        print("Using file logger for {}".format(logfile))
        log_handler = logging.FileHandler(logfile)
        log_handler.setLevel(logging.DEBUG)
        log_handler.setFormatter(log_formatter)
        log.addHandler(log_handler)

    log.info('=' * 75)
    log.info(('=' * 10) + (' {} '.format(app_name)) + ('version {} '.format(__version__)))
    log.info('=' * 75)

    return log
