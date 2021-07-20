# -*- coding: utf-8 -*-
"""
comparison_qrs_detectors.download
---------------
This module provides several data download methods for scientific databases from https://physionet.org.

:copyright: (c) 2021 by Open Innovation Lab
:license: BSD 3-clause, see LICENSE for more details.
"""

# Imports
# built-in
import os

# 3rd party
import requests
import wget


def download_url(url, root, filename=None):
    """Download a file from a url and place it in root.
    Args:
        url (str): URL to download file from
        root (str): Directory to place downloaded file in
        filename (str, optional): Name to save the file under. If None, use the basename of the URL
    """
    root = os.path.expanduser(root)
    if not filename:
        filename = os.path.basename(url)
    fpath = os.path.join(root, filename)

    print(fpath)

    # create directory if not already existing
    os.makedirs(root, exist_ok=True)
    print(root)

    # try to download files
    try:
        print('Downloading ' + url + ' to ' + fpath)
        wget.download(url, out=fpath)
        print('Response OK')
    except:
        try:
            if url[:5] == 'https':
                url = url.replace('https:', 'http:')
                print('Failed download. Trying https -> http instead.'
                      ' Downloading ' + url + ' to ' + fpath)
                wget.download(url, out=fpath)
                print('Response OK')
        except:
            print('Response FAILED')

#import subprocess
#command = ["C:/Users/Andreas/GnuWin32/bin/wget.exe", "-r", "-c", "-np","https://physionet.org/files/ludb/1.0.0/", "-o", "C:/Users/Andreas/PycharmProjects/Comparison-QRS-Detectors/data/raw", "--no-check-certificate"]
#process = subprocess.Popen(command)
