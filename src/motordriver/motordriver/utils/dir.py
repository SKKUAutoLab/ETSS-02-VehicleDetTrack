# ==================================================================== #
# File name: dir.py
# Author: Automation Lab - Sungkyunkwan University
# Date created: 03/27/2021
# ==================================================================== #
import os
from glob import glob
from typing import List

""" 
File, dir, Path definitions:
- drive         : a string that represents the drive name. For example, PureWindowsPath("c:/Program Files/CSV").drive returns "C:"
- parts         : returns a tuple that provides access to the path"s components
- name          : the path component without any dir
- parent        : sequence providing access to the logical ancestors of the path
- stem          : final path component without its suffix
- suffix        : the file extension of the final component
- anchor        : the part of a path before the dir. / is used to create child paths and mimics the behavior of os.path.join.
- joinpath      : combines the path with the arguments provided
- match(pattern): returns True/False, based on matching the path with the glob-style pattern provided 

In path "/home/mains/stackabuse/python/sample.md":
- path              : - returns PosixPath("/home/mains/stackabuse/python/sample.md")
- path.parts        : - returns ("/", "home", "mains", "stackabuse", "python")
- path.name         : - returns "sample.md"
- path.stem         : - returns "sample"
- path.suffix       : - returns ".md"
- path.parent       : - returns PosixPath("/home/mains/stackabuse/python")
- path.parent.parent: - returns PosixPath("/home/mains/stackabuse")
- path.match("*.md"): returns True
- PurePosixPath("/python").joinpath("edited_version"): returns ("home/mains/stackabuse/python/edited_version
"""

# NOTE: Inside TSS//tss/utils
utils_dir    = os.path.dirname(os.path.abspath(__file__))  # "workspaces/workspace/TSS/tss/utils"
tss_dir      = os.path.dirname(utils_dir)                  # "workspaces/workspace/TSS/tss"

root_dir     = os.path.dirname(tss_dir)                    # "workspaces/workspace/TSS"
bin_dir      = os.path.join(root_dir, "mains")             # "workspaces/workspace/TSS/bin"
data_dir     = os.path.join(root_dir, "data")              # "workspaces/workspace/TSS/data"
tests_dir    = os.path.join(root_dir, "tests")             # "workspaces/workspace/TSS/tests"


# MARK: Create

def create_dirs(dirs: List[str], delete_existing: bool = False):
    """Check and create directories if they are not found.
    """
    try:
        for d in dirs:
            if os.path.exists(d) and delete_existing:
                files = glob(d + "/*")
                for f in files:
                    os.remove(f)
            elif not os.path.exists(d):
                os.makedirs(d)
        return 0
    except Exception as err:
        print("[ERROR] Creating directories error: {0}".format(err))
        exit(-1)


# MARK: Read

def list_subdirs(current_dir: str) -> List[str]:
    """List all subdirectories.
    """
    return [d for d in os.listdir(current_dir) if os.path.isdir(os.path.join(current_dir, d))]


def list_files(patterns: List[str]) -> List[str]:
    """List all files that match the desired extension.
    """
    image_paths = []
    for pattern in patterns:
        absolute_paths = glob(pattern)
        for abs_path in absolute_paths:
            if os.path.isfile(abs_path):
                image_paths.append(abs_path)
    return image_paths


# MARK: Update


# MARK: Delete

def delete_files(dirs: List[str], extension: str = "", recursive: bool = True):
    """Delete all files in directories that match the desired extension.
    """
    for d in dirs:
        if os.path.exists(d):
            files = glob(d + "/*" + extension, recursive=recursive)
            for f in files:
                os.remove(f)


def delete_files_matching(patterns: List[str]):
    """Delete all files that match the desired patterns.
    """
    for pattern in patterns:
        files = glob(pattern)
        for f in files:
            os.remove(f)
