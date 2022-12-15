# ==================================================================== #
# Copyright (C) 2022 - Automation Lab - Sungkyunkwan University
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU General Public License
# as published by the Free Software Foundation; either version 2
# of the License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program; if not, write to the Free Software
# Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301, USA.
#
# ``Tracker`` base class for all variant of tracker.
# It define a unify template to guarantee the input and output of all tracker are the same.
# Usually, each ``Tracker`` class is associate with a ``Track`` class
#
# Subclassing guide:
# 1. The package (i.e, the .py filename) should be in the template:
#    {tracker}_{track_motion_model}_{feature_used_to_track}
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
