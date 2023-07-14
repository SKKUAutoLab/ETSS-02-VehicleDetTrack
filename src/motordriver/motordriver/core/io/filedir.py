#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Functions for managing files and directories.

File, dir, Path definitions:
- drive         : A string that represents the drive name. For example,
                  PureWindowsPath("c:/Program Files/CSV").drive returns "C:"
- parts         : Return a tuple that provides access to the path"s components
- name          : Path component without any dir.
- parent        : sequence providing access to the logical ancestors of the
                  path.
- stem          : Final path component without its suffix.
- suffix        : the file extension of the final component.
- anchor        : Part of a path before the dir. / is used to create child
                  paths and mimics the behavior of os.path.join.
- joinpath      : Combine the path with the arguments provided.
- match(pattern): Return True/False, based on matching the path with the
                  glob-style pattern provided

Working with Path:
path = Path("/home/mains/stackabuse/python/sample.md")
- path              : Return PosixPath("/home/mains/stackabuse/python/sample.md")
- path.parts        : Return ("/", "home", "mains", "stackabuse", "python")
- path.name         : Return "sample.md"
- path.stem         : Return "sample"
- path.suffix       : Return ".md"
- path.parent       : Return PosixPath("/home/mains/stackabuse/python")
- path.parent.parent: Return PosixPath("/home/mains/stackabuse")
- path.match("*.md"): Return True
- PurePosixPath("/python").joinpath("edited_version"): returns ("home/mains/stackabuse/python/edited_version)
"""

from __future__ import annotations

import os
import shutil
from glob import glob
from pathlib import Path
from typing import Optional

import validators

from core.type.type import ScalarListOrTupleAnyT
from core.type.collection import unique
from core.utils.rich import console

__all__ = [
    "create_dirs",
    "delete_files",
    "delete_files_matching",
    "get_dirname",
    "get_hash",
    "get_latest_file",
    "get_stem",
    "has_subdir",
    "is_basename",
    "is_ckpt_file",
    "is_json_file",
    "is_name",
    "is_stem",
    "is_torch_saved_file",
    "is_txt_file",
    "is_url",
    "is_url_or_file",
    "is_weights_file",
    "is_xml_file",
    "is_yaml_file",
    "list_files",
    "list_subdirs"
]


# MARK: - Create

def create_dirs(paths: ScalarListOrTupleAnyT[str], recreate: bool = False):
    """Check and create directories.

    Args:
        paths (ScalarListOrTupleAnyT[str]):
            List of directories' paths to create.
        recreate (bool):
            If `True`, delete and recreate existing directories.
    """
    if isinstance(paths, str):
        paths = [paths]
    elif isinstance(paths, tuple):
        paths = list(paths)

    unique_dirs = unique(paths)
    try:
        for d in unique_dirs:
            if os.path.exists(d) and recreate:
                shutil.rmtree(d)
            if not os.path.exists(d):
                os.makedirs(d)
        return 0
    except Exception as err:
        console.log(f"Cannot create directory: {err}.")
        # exit(-1)
        
        
# MARK: - Validate

def has_subdir(path: str, name: str) -> bool:
    """Return `True` if the subdirectory with `name` is found inside `path`."""
    subdirs = list_subdirs(path=path)
    return name in subdirs


def is_basename(path: Optional[str]) -> bool:
    """Check if the given path is a basename."""
    if path is None:
        return False

    parent = str(Path(path).parent)
    if parent == ".":
        root, ext = os.path.splitext(path)
        if ext != "":
            return True

    return False


def is_ckpt_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.ckpt` file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".ckpt"]:
            return True

    return False


def is_json_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.json` file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".json"]:
            return True

    return False


def is_name(path: Optional[str]) -> bool:
    """Check if the given path is a name."""
    if path is None:
        return False

    name = str(Path(path.lower()).name)
    if name == path:
        return True

    return False


def is_stem(path: Optional[str]) -> bool:
    """Check if the given path is a stem."""
    if path is None:
        return False

    path   = path.lower()
    parent = str(Path(path).parent)
    if parent == ".":
        root, ext = os.path.splitext(path)
        if ext == "":
            return True

    return False


def is_torch_saved_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.pt`, `.pth`, `.weights`, or `.ckpt` file.
    """
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".pt", ".pth", ".weights", ".ckpt"]:
            return True

    return False


def is_txt_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.txt` file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".txt"]:
            return True

    return False


def is_url(path: Optional[str]) -> bool:
    """Check if the given path is a valid url."""
    if path is None:
        return False

    if isinstance(validators.url(path), validators.ValidationFailure):
        return False
    return True


def is_url_or_file(path: Optional[str]) -> bool:
    """Check if the given path is a valid url or a local file."""
    if path is None:
        return False
    
    if os.path.isfile(path=path):
        return True
    if isinstance(validators.url(path), validators.ValidationFailure):
        return False
    return True


def is_weights_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.pt` or `.pth` file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".pt", ".pth"]:
            return True

    return False


def is_xml_file(path: Optional[str]) -> bool:
    """Check if the given path is a .xml file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".xml"]:
            return True

    return False


def is_yaml_file(path: Optional[str]) -> bool:
    """Check if the given path is a `.yaml` file."""
    if path is None:
        return False

    if os.path.isfile(path=path):
        extension = os.path.splitext(path.lower())[1]
        if extension in [".yaml", ".yml"]:
            return True

    return False


# MARK: - Read

def get_stem(path: str) -> str:
    """Get the stem from the given path."""
    basename = os.path.basename(path)
    stem, _  = os.path.splitext(basename)
    return stem


get_dirname = get_stem


def get_latest_file(path: str, recursive: bool = True) -> Optional[str]:
    """Get the latest file from a folder or pattern according to modified time.

    Args:
        path (str):
            Directory path or path pattern.
        recursive (str):
            Should look for sub-directories also?.

    Returns:
        latest_path (str, optional):
            Latest file path. Return `None` if not found (no file, wrong path
            format, wrong file extension).
    """
    file_list = glob(path, recursive=recursive)
    if len(file_list) > 0:
        return max(file_list, key=os.path.getctime)
    return None


def get_hash(files: ScalarListOrTupleAnyT[str]):
    """Returns a single hash value of a list of files."""
    if isinstance(files, str):
        files = [files]
    elif isinstance(files, tuple):
        files = list(files)
    return sum(os.path.getsize(f) for f in files if os.path.isfile(f))


def list_files(patterns: ScalarListOrTupleAnyT[str]) -> list[str]:
    """List all files that match the desired patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    elif isinstance(patterns, tuple):
        patterns = list(patterns)
        
    image_paths = []
    for pattern in patterns:
        absolute_paths = glob(pattern)
        for abs_path in absolute_paths:
            if os.path.isfile(abs_path):
                image_paths.append(abs_path)
    return image_paths


def list_subdirs(path: Optional[str]) -> Optional[list[str]]:
    """List all subdirectories inside the given `path`."""
    if path is None:
        return None

    return [
        d for d in os.listdir(path) if os.path.isdir(os.path.join(path, d))
    ]


# MARK: - Delete

def delete_files(
    dirpaths : ScalarListOrTupleAnyT[str],
    extension: str = "",
    recursive: bool = True
):
    """Delete all files in directories that match the desired extension.

    Args:
        dirpaths: (ScalarListOrTupleAnyT[str])
            List of directories' paths that contains the files to be deleted.
        extension (str):
            File extension. Default: "".
        recursive (bool):
            Search subdirectories if any. Default: `True`.
    """
    if isinstance(dirpaths, str):
        dirpaths = [dirpaths]
    elif isinstance(dirpaths, tuple):
        dirpaths = list(dirpaths)
        
    unique_dirs = unique(dirpaths)
    extension   = f".{extension}" if "." not in extension else extension
    for d in unique_dirs:
        # if os.path.exists(d):
        pattern = os.path.join(d, f"*{extension}")
        files   = glob(pattern, recursive=recursive)
        for f in files:
            console.log(f"Deleting {f}.")
            os.remove(f)


def delete_files_matching(patterns: ScalarListOrTupleAnyT[str]):
    """Delete all files that match the desired patterns."""
    if isinstance(patterns, str):
        patterns = [patterns]
    elif isinstance(patterns, tuple):
        patterns = list(patterns)
        
    for pattern in patterns:
        files = glob(pattern)
        for f in files:
            os.remove(f)
