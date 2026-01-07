#!/usr/bin/env python3
import argparse
import hashlib
import os
import sys
import tempfile
import shutil
import urllib.request


def sha256_of_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def download(url, dest, expected_sha256=None):
    tmp_fd, tmp_path = tempfile.mkstemp(suffix='.download')
    os.close(tmp_fd)
    try:
        print(f'Downloading {url} ...')
        urllib.request.urlretrieve(url, tmp_path)
        if expected_sha256:
            actual = sha256_of_file(tmp_path)
            if actual.lower() != expected_sha256.lower():
                raise RuntimeError(f"SHA256 mismatch: expected {expected_sha256}, got {actual}")
        shutil.move(tmp_path, dest)
        print(f'Saved model to {dest}')
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except Exception:
                pass


def main():
    p = argparse.ArgumentParser(description='Download ai_model.h5 into project root')
    p.add_argument('url', nargs='?', help='Model URL (or set MODEL_URL env var)')
    p.add_argument('--sha256', help='Optional expected SHA256 checksum')
    p.add_argument('--dest', default='ai_model.h5', help='Destination filename')
    args = p.parse_args()
    url = args.url or os.environ.get('MODEL_URL')
    if not url:
        print('No model URL provided. Provide it as argument or set MODEL_URL env var.', file=sys.stderr)
        sys.exit(2)
    try:
        download(url, args.dest, args.sha256)
    except Exception as e:
        print('Download failed:', e, file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
