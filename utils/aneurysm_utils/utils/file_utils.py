"""Utilities for file handler operation."""

import atexit
import logging
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from datetime import date, datetime
from typing import List, Optional, Tuple

import requests
import tqdm

from aneurysm_utils.utils import request_utils, system_utils

log = logging.getLogger(__name__)


def folder_size(path: str) -> str:
    """Disk usage of a specified folder."""
    return (
        subprocess.check_output(["du", "-sh", "-B1", path]).split()[0].decode("utf-8")
    )


def is_subdir(path: str, directory: str) -> bool:
    """Checks if a `directory` is the subdirectory for a given `path`."""
    path = os.path.realpath(path)
    directory = os.path.realpath(directory)
    relative = os.path.relpath(path, directory)
    return not (relative == os.pardir or relative.startswith(os.pardir + os.sep))


def remove_folder_content(folder):
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)


def identify_compression(file_path: str) -> Optional[str]:
    """Indentifies the compression type of the given file."""
    sign_dict = {
        b"\x1f\x8b\x08": "gz",
        b"\x42\x5a\x68": "bz2",
        b"\x50\x4b\x03\x04": "zip",
        b"\x37\x7a\xbc\xaf\x27\x1c": "7z",
        b"\x75\x73\x74\x61\x72": "tar",
        b"\x52\x61\x72\x21\x1a\x07\x00": "rar",
    }

    max_len = max(len(x) for x in sign_dict)
    with open(file_path, "rb") as f:
        file_start = f.read(max_len)
    for magic, filetype in sign_dict.items():
        if file_start.startswith(magic):
            return filetype
    return None


def get_last_usage_date(path: str) -> datetime:
    """Returns last usage date for a given file."""
    date: datetime

    if not os.path.exists(path):
        raise FileNotFoundError("Path does not exist: " + path)

    try:
        date = datetime.fromtimestamp(os.path.getmtime(path))
    except Exception:
        pass

    try:
        compare_date = datetime.fromtimestamp(os.path.getatime(path))
        if date.date() < compare_date.date():
            # compare date is newer
            date = compare_date
    except Exception:
        pass

    try:
        compare_date = datetime.fromtimestamp(os.path.getctime(path))
        if date.date() < compare_date.date():
            # compare date is newer
            date = compare_date
    except Exception:
        pass

    return date


def extract_zip(
    file_path: str, unpack_path: str = None, remove_if_exists: bool = False
) -> Optional[str]:
    """
    Unzips a file.

    Args:
        file_path: Path to zipped file.
        unpack_path: Path to unpack the file (optional).
        remove_if_exists: If `True`, the directory will be removed if it already exists (optional).

    Returns:
        Path to the unpacked folder or `None` if unpacking failed.
    """
    if not os.path.exists(file_path):
        log.warning(file_path + " does not exist.")
        return None

    if not zipfile.is_zipfile(file_path):
        log.warning(file_path + " is not a zip file.")
        return None

    if not unpack_path:
        unpack_path = os.path.join(
            os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]
        )

    if os.path.isdir(unpack_path):
        log.info("Unpack directory already exists " + unpack_path)

        if not os.listdir(unpack_path):
            log.info("Directory is empty. Unpacking...")
            shutil.rmtree(unpack_path)
        elif remove_if_exists:
            log.info("Removing existing unpacked dir: " + unpack_path)
            shutil.rmtree(unpack_path)
        else:
            return unpack_path

    log.info("Unpacking file " + os.path.basename(file_path) + " to: " + unpack_path)
    zip_ref = zipfile.ZipFile(file_path, "r")
    zip_ref.extractall(unpack_path)
    zip_ref.close()

    if not os.path.exists(unpack_path):
        log.warning("Failed to extract zip file: " + file_path)

    return unpack_path


def zip_folder(
    folder_path: str,
    archive_file_name: str = None,
    max_file_size: int = None,
    excluded_folders: List[str] = None,
    compression: int = zipfile.ZIP_STORED,
) -> Optional[str]:
    """
    Zips a folder (via `zipfile`). The folder will be zipped to an archive file in a temp directory.

    Args:
        folder_path: Path to the folder.
        archive_file_name: Name of the resulting zip package file (optional).
        max_file_size: Max file size in `MB` to be included in the archive (optional).
        excluded_folders: List of folders to exclude from the archive (optional).
        compression: Compression mode. Please see the `zipfile` documentation for supported compression modes (optional).

    Returns:
        Path to the zipped archive file or `None` if zipping failed.
    """
    # TODO accept names with wildcards in exclude like for tar
    if not os.path.isdir(folder_path):
        log.info("Failed to zip (not a directory): " + folder_path)
        return None

    temp_folder = tempfile.mkdtemp()

    if max_file_size:
        max_file_size = max_file_size * 1000000  # MB ro bytes

    def cleanup():
        log.info("Removing temp directory: " + temp_folder)
        shutil.rmtree(temp_folder)

    atexit.register(cleanup)

    if not archive_file_name:
        archive_file_name = os.path.basename(folder_path) + ".zip"

    zip_file_path = os.path.join(temp_folder, archive_file_name)
    log.debug("Zipping folder: " + folder_path + " to " + zip_file_path)
    zip_file = zipfile.ZipFile(zip_file_path, "w", compression)

    # dont packge folder inside, only package everything inside folder
    for dirname, subdirs, files in os.walk(folder_path):
        if excluded_folders:
            for excluded_folder in excluded_folders:
                if excluded_folder in subdirs:
                    log.debug("Ignoring folder because of name: " + excluded_folder)
                    subdirs.remove(excluded_folder)
        if dirname != folder_path:
            # only write if dirname is not the root folder
            zip_file.write(dirname, os.path.relpath(dirname, folder_path))
        for filename in files:
            if max_file_size and max_file_size < os.path.getsize(
                os.path.join(dirname, filename)
            ):
                # do not write file if it is bigger than
                log.debug("Ignoring file because of file size: " + filename)
                continue
            file_path = os.path.join(dirname, filename)
            zip_file.write(file_path, os.path.relpath(file_path, folder_path))
    zip_file.close()

    return zip_file_path


def tar_folder(
    folder_path: str,
    archive_file_name: str = None,
    max_file_size: int = None,
    exclude: List[str] = None,
    compression: bool = False,
) -> Optional[str]:
    """
    Tars a folder (via tar). The folder will be packaged to an archive file in a temp directory.

    Args:
        folder_path: Path to the folder.
        archive_file_name: Name of the resulting tar package file (optional).
        max_file_size: Max file size in `MB` to be included in the archive (optional).
        exclude: List of files or folders to exclude from the archive. This also supports wildcards (optional).
        compression: If `True`, compression will be applied (optional).

    Returns:
        Path to the packaged archive file or `None` if tar-process failed.
    """
    if not os.path.isdir(folder_path):
        log.info("Failed to package to tar (not a directory): " + folder_path)
        return None

    temp_folder = tempfile.mkdtemp()

    def cleanup():
        log.info("Removing temp directory: " + temp_folder)
        shutil.rmtree(temp_folder)

    atexit.register(cleanup)

    if not archive_file_name:
        archive_file_name = os.path.basename(folder_path) + ".tar"

    archive_file_path = os.path.join(temp_folder, archive_file_name)
    tar_options = " --ignore-failed-read "
    if max_file_size:
        tar_options += (
            " --exclude-from <(find '"
            + folder_path
            + "' -size +"
            + str(max_file_size)
            + "M)"
        )
    if exclude:
        for excluded in exclude:
            tar_options += " --exclude='" + excluded + "' "

    tar_mode = " -cf "  # no compression
    if compression:
        tar_mode = " -czf "

    log.debug("Packaging (via tar) folder: " + folder_path + " to " + archive_file_path)

    tar_command = (
        "tar "
        + tar_options
        + tar_mode
        + " '"
        + archive_file_path
        + "' -C '"
        + folder_path
        + "' ."
    )
    log.info("Executing: " + tar_command)
    # exclude only works with bash
    exit_code = system_utils.bash_command(tar_command)
    log.info("Finished with exit code: " + str(exit_code))

    # TODO check call return if successful
    if not os.path.isfile(archive_file_path):
        log.warning("Failed to tar folder: " + archive_file_path)

    return archive_file_path


def extract_tar(
    file_path: str, unpack_path: str = None, remove_if_exists: bool = False
) -> Optional[str]:
    """
    Extracts a tar file.

    Args:
        file_path: Path to tar file.
        unpack_path: Path to unpack the file (optional).
        remove_if_exists: If `True`, the directory will be removed if it already exists (optional).

    Returns:
        Path to the unpacked folder or `None` if unpacking failed.
    """

    if not os.path.exists(file_path):
        log.warning(file_path + " does not exist.")
        return None

    if not tarfile.is_tarfile(file_path):
        log.warning(file_path + " is not a tar file.")
        return None

    if not unpack_path:
        unpack_path = os.path.join(
            os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]
        )

    if os.path.isdir(unpack_path):
        log.info("Unpack directory already exists " + unpack_path)
        if not os.listdir(unpack_path):
            log.info("Directory is empty. Unpacking...")
        elif remove_if_exists:
            log.info("Removing existing unpacked dir: " + unpack_path)
            shutil.rmtree(unpack_path)
        else:
            return unpack_path

    log.info("Unpacking file " + os.path.basename(file_path) + " to: " + unpack_path)
    compression = identify_compression(file_path)
    if not compression:
        mode = "r"
    elif compression == "gz":
        mode = "r:gz"
    elif compression == "bz2":
        mode = "r:bz2"
    else:
        mode = "r"

    tar = tarfile.open(file_path, mode)
    tar.extractall(unpack_path)
    tar.close()

    # Tar unpacking via tar command
    # tar needs empty directory
    # if not os.path.exists(unpack_path):
    #    os.makedirs(unpack_path)
    # log.info("Unpacking (via tar command) file " + os.path.basename(file_path) + " to: " + unpack_path)
    # handle compression with -zvxf
    # cmd = "tar -xf " + file_path + " -C " + unpack_path
    # log.debug("Executing: " + cmd)
    # exit_code = system_utils.bash_command(cmd)
    # log.info("Finished with exit code: " + str(exit_code))

    if not os.path.exists(unpack_path):
        log.warning("Failed to extract tar file: " + file_path)

    return unpack_path


def extract_via_patoolib(
    file_path: str, unpack_path: str = None, remove_if_exists: bool = False
) -> Optional[str]:
    """
    Extracts an archive file via patoolib.

    Args:
        file_path: Path to an archive file.
        unpack_path: Path to unpack the file (optional).
        remove_if_exists: If `True`, the directory will be removed if it already exists (optional).

    Returns:
        Path to the unpacked folder or `None` if unpacking failed.
    """
    # TODO handle compression with -zvxf
    if not os.path.exists(file_path):
        log.warning(file_path + " does not exist.")
        return None

    try:
        import patoolib
    except ImportError:
        log.warning("patoolib is not installed: Run pip install patool")
        return None

    if not unpack_path:
        unpack_path = os.path.join(
            os.path.dirname(file_path), os.path.splitext(os.path.basename(file_path))[0]
        )

    if os.path.isdir(unpack_path):
        log.info("Unpack directory already exists " + unpack_path)
        if not os.listdir(unpack_path):
            log.info("Directory is empty. Unpacking...")
        elif remove_if_exists:
            log.info("Removing existing unpacked dir: " + unpack_path)
            shutil.rmtree(unpack_path)
        else:
            return unpack_path

    try:
        patoolib.extract_archive(file_path, outdir=unpack_path)
    except Exception as e:
        log.warning("Failed to unpack via patoolib: ", exc_info=e)
        return None

    return unpack_path


def download_file(
    url: str, folder_path: str, file_name: str = None, force_download: bool = False
) -> Tuple[str, bool]:
    """
    Downloads a file from an URL.

    Args:
        url: Path to an archive file.
        folder_path: Folder to download the file to (optional).
        file_name: Filename to use for the downloaded file (optional).
        force_download: If `True`, the file will always be downloaded even if it already exists locally (optional).

    Returns:
        A Tuple of the path to the downloaded file and `True` if file was downloaded.
    """
    # NOTE the str eam=True parameter below
    with requests.get(url, stream=True, allow_redirects=True) as r:
        if r.status_code >= 400:
            log.warning(
                "The request was not successfull (status code: "
                + str(r.status_code)
                + ")"
            )

            if file_name and os.path.exists(os.path.join(folder_path, file_name)):
                return os.path.join(folder_path, file_name), False
            else:
                return None, False

        content_type = r.headers.get("Content-Type")
        if content_type and "html" in content_type.lower():
            log.warning(
                "The url is pointing to an HTML page. Are you sure you want to download this as file?"
            )
        try:
            total_length = int(r.headers.get("Content-Disposition").split("size=")[1])
        except Exception:
            try:
                total_length = int(r.headers.get("Content-Length"))
            except Exception:
                log.warning("Failed to figure out size of file.")
                total_length = 0

        if not file_name:
            # if file name is not provided use filename from url
            file_name = request_utils.url2filename(url)  # url.split('/')[-1]
            try:
                # Try to use filename from content disposition
                file_name = request_utils.get_filename_from_cd(
                    r.headers.get("Content-Disposition")
                )
            except Exception:
                pass

        if not file_name:
            log.warning("No file name was determined.")
            file_name = str(date.today()) + "-download"

        file_path = os.path.join(folder_path, file_name)
        if not force_download and os.path.isfile(file_path):
            if total_length == os.path.getsize(file_path):
                log.info(
                    "File "
                    + file_name
                    + " already exists with same size and will not be downloaded."
                )
                # file already exists and has same size -> do not download
                return file_path, False

        if not os.path.exists(os.path.dirname(file_path)):
            os.makedirs(os.path.dirname(file_path))

        with open(file_path, "wb") as f:
            pbar = tqdm.tqdm(
                total=total_length,
                initial=0,
                mininterval=0.3,
                unit="B",
                unit_scale=True,
                desc="Downloading " + str(file_name),
                file=sys.stdout,
            )
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:  # filter out keep-alive new chunks
                    pbar.update(len(chunk))
                    f.write(chunk)
            pbar.close()
        return file_path, True


def unpack_archive(
    file_path: str, unpack_path: str = None, remove_if_exists: bool = False
) -> Optional[str]:
    """
    Unpacks a compressed file.

    Args:
        file_path: Path to compressed file.
        unpack_path: Path to unpack the file (optional).
        remove_if_exists: If `True`, the directory will be removed if it already exists (optional).

    Returns:
        Path to the unpacked folder or `None` if unpacking failed.
    """
    import tarfile
    import zipfile

    if not os.path.isfile(file_path):
        log.warning("File does not exist: " + file_path)
        return None

    if zipfile.is_zipfile(file_path):
        unpack_path = extract_zip(file_path, unpack_path, remove_if_exists)
    elif tarfile.is_tarfile(file_path):
        unpack_path = extract_tar(file_path, unpack_path, remove_if_exists)
    else:
        unpack_path = extract_via_patoolib(file_path, unpack_path, remove_if_exists)

    if unpack_path and os.path.isdir(unpack_path):
        unpack_folder_name = os.path.basename(unpack_path)
        if len(os.listdir(unpack_path)) == 1 and unpack_folder_name in os.listdir(
            unpack_path
        ):
            # unpacked folder contains one folder with same name -> move content to higher up folder
            folder_to_move = os.path.join(unpack_path, unpack_folder_name)
            files = os.listdir(folder_to_move)
            for f in files:
                shutil.move(os.path.join(folder_to_move, f), unpack_path)

            # Remove empty folder
            if len(os.listdir(folder_to_move)) == 0:
                os.rmdir(folder_to_move)
            else:
                log.info("Folder content was moved but folder is not empty.")

    return unpack_path


def cleanup_folder(
    folder_path: str,
    max_file_size_mb: int = 50,
    last_file_usage: int = 3,
    replace_with_info: bool = True,
    excluded_folders: List[str] = None,
):
    """
    Cleans up a folder to reduce disk space usage.

    Args:
        folder_path: Folder that should be cleaned.
        max_file_size_mb: Max size of files in MB that should be deleted. Default: 50.
        replace_with_info: Replace removed files with `.removed.txt` files with file removal reason. Default: True.
        last_file_usage: Number of days a file wasn't used to allow the file to be removed. Default: 3.
        excluded_folders: List of folders to exclude from removal (optional).
    """
    total_cleaned_up_mb = 0
    removed_files = 0

    for dirname, subdirs, files in os.walk(folder_path):
        if excluded_folders:
            for excluded_folder in excluded_folders:
                if excluded_folder in subdirs:
                    log.debug("Ignoring folder because of name: " + excluded_folder)
                    subdirs.remove(excluded_folder)
        for filename in files:
            file_path = os.path.join(dirname, filename)

            file_size_mb = int(os.path.getsize(file_path) / (1024.0 * 1024.0))
            if max_file_size_mb and max_file_size_mb > file_size_mb:
                # File will not be deleted since it is less than the max size
                continue

            last_file_usage_days = None
            if get_last_usage_date(file_path):
                last_file_usage_days = (
                    datetime.now() - get_last_usage_date(file_path)
                ).days

            if last_file_usage_days and last_file_usage_days <= last_file_usage:
                continue

            current_date_str = datetime.now().strftime("%B %d, %Y")
            removal_reason = (
                "File has been removed during folder cleaning ("
                + folder_path
                + ") on "
                + current_date_str
                + ". "
            )
            if file_size_mb and max_file_size_mb:
                removal_reason += (
                    "The file size was "
                    + str(file_size_mb)
                    + " MB (max "
                    + str(max_file_size_mb)
                    + "). "
                )

            if last_file_usage_days and last_file_usage:
                removal_reason += (
                    "The last usage was "
                    + str(last_file_usage_days)
                    + " days ago (max "
                    + str(last_file_usage)
                    + "). "
                )

            log.info(filename + ": " + removal_reason)

            # Remove file
            try:
                os.remove(file_path)

                if replace_with_info:
                    with open(file_path + ".removed.txt", "w") as file:
                        file.write(removal_reason)

                if file_size_mb:
                    total_cleaned_up_mb += file_size_mb

                removed_files += 1

            except Exception as e:
                log.info("Failed to remove file: " + file_path, e)

    log.info(
        "Finished cleaning. Removed "
        + str(removed_files)
        + " files with a total disk space of "
        + str(total_cleaned_up_mb)
        + " MB."
    )
