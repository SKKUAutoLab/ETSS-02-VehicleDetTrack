# ==================================================================== #
# File name: print.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/18/2021
# ==================================================================== #
from sty import bg, ef, fg, rs
from sty import RgbFg, Style


# MARK: - Print

def prints(s: str):
    """Print normal status.
    """
    s = fg.li_green + s + fg.rs
    print(s)


def printw(s: str):
    """Print warning.
    """
    s = fg.li_yellow + s + fg.rs
    print(s)


def printe(s: str):
    """PrintError.
    """
    s = fg.li_red + s + fg.rs
    print(s)
