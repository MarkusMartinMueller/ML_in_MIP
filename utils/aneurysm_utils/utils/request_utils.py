"""Utilities for HTTP request operations."""

import os
import posixpath
import re

from typing import Optional

try:
    from urlparse import urlsplit
    from urllib import unquote
except ImportError:  # Python 3
    from urllib.parse import urlsplit, unquote

url_validator = re.compile(
    r"^(?:http|ftp)s?://"  # http:// or https://
    # domain...
    r"(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+(?:[A-Z]{2,6}\.?|[A-Z0-9-]{2,}\.?)|"
    r"localhost|"  # localhost...
    r"\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})"  # ...or ip
    r"(?::\d+)?"  # optional port
    r"(?:/?|[/?]\S+)$",
    re.IGNORECASE,
)


def is_valid_url(url: str) -> bool:
    """Check is the provided URL is valid."""
    return re.match(url_validator, url) is not None


def is_downloadable(url: str) -> bool:
    """
    Does the url contain a downloadable resource
    """
    if not is_valid_url(url):
        return False

    try:
        import requests

        h = requests.head(url, allow_redirects=True)
        if h.status_code >= 400:
            # request not successfull
            return False
        header = h.headers
        content_type = header.get("content-type")
        if content_type and "html" in content_type.lower():
            return False
        return True
    except Exception:
        return False


def url2filename(url: str) -> str:
    """Return basename corresponding to url.
    >>> print(url2filename('http://example.com/path/to/file%C3%80?opt=1'))
    fileÀ
    >>> print(url2filename('http://example.com/slash%2fname')) # '/' in name
    Traceback (most recent call last):
    ...
    ValueError
    """
    urlpath = urlsplit(url).path
    basename = posixpath.basename(unquote(urlpath))
    if (
        os.path.basename(basename) != basename
        or unquote(posixpath.basename(urlpath)) != basename
    ):
        raise ValueError  # reject '%2f' or 'dir%5Cbasename.ext' on Windows
    return basename


def get_filename_from_cd(content_disposition: str) -> Optional[str]:
    """Returns the filename from content-disposition."""
    # TODO: filename\*?=([^;]+)?
    fname_list = re.findall(
        "filename\*=([^;]+)", content_disposition, flags=re.IGNORECASE
    )
    if not fname_list:
        fname_list = re.findall(
            "filename=([^;]+)", content_disposition, flags=re.IGNORECASE
        )

    if not fname_list:
        return None

    if fname_list[0].lower().startswith("utf-8''"):
        fname = re.sub("utf-8''", "", fname_list[0], flags=re.IGNORECASE)
        fname = unquote(fname)
    else:
        fname = fname_list[0]
    # clean space and double quotes
    return fname.strip().strip('"')
