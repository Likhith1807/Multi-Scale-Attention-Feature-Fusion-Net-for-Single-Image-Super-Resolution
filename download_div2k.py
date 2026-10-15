"""Download official DIV2K training/validation HR and bicubic x4 LR archives."""
import argparse
from pathlib import Path, PurePosixPath
import shutil
import time
import urllib.request
import zipfile


BASE = "https://data.vision.ee.ethz.ch/cvl/DIV2K/"
ARCHIVES = [
    ("DIV2K_train_HR.zip", "DIV2K_train_HR", 800),
    ("DIV2K_train_LR_bicubic_X4.zip", "DIV2K_train_LR_bicubic/X4", 800),
    ("DIV2K_valid_HR.zip", "DIV2K_valid_HR", 100),
    ("DIV2K_valid_LR_bicubic_X4.zip", "DIV2K_valid_LR_bicubic/X4", 100),
]


def download(url, destination):
    destination = Path(destination)
    if destination.is_file():
        if not zipfile.is_zipfile(destination):
            raise ValueError(f"Existing file is not a ZIP: {destination}")
        return
    partial = destination.with_suffix(".zip.part")
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "SISR-Research-Starter/0.2"}
    if offset:
        headers["Range"] = f"bytes={offset}-"
    request = urllib.request.Request(url, headers=headers)
    print(f"Downloading {destination.name}; this may take time.", flush=True)
    with urllib.request.urlopen(request, timeout=60) as response:
        append = offset > 0 and response.status == 206
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise RuntimeError("Download range mismatch. Keep the partial file and retry.")
        if not append:
            offset = 0
        content_length = response.headers.get("Content-Length")
        expected = offset + int(content_length) if content_length else None
        received, last_report = offset, time.monotonic()
        with partial.open("ab" if append else "wb") as file:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                file.write(chunk)
                received += len(chunk)
                if time.monotonic() - last_report >= 5:
                    if expected:
                        print(f"  {received / 1024 ** 2:.0f} / "
                              f"{expected / 1024 ** 2:.0f} MiB "
                              f"({100 * received / expected:.1f}%)", flush=True)
                    else:
                        print(f"  {received / 1024 ** 2:.0f} MiB received "
                              "(server did not provide total size)", flush=True)
                    last_report = time.monotonic()
    if expected is not None and received != expected:
        raise RuntimeError("Download incomplete. Rerun to resume the .part file.")
    if not zipfile.is_zipfile(partial):
        raise ValueError("Downloaded response is not a valid ZIP.")
    partial.replace(destination)


def extract_pngs(archive, root):
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename.replace("\\", "/")
            parts = PurePosixPath(name)
            if parts.is_absolute() or ".." in parts.parts or ":" in name:
                raise ValueError(f"Unsafe archive path: {info.filename}")
            target = (root / name).resolve()
            if not target.is_relative_to(root):
                raise ValueError("Archive path escapes the data directory.")
            if info.is_dir() or target.suffix.lower() != ".png":
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".png.part")
            # Reading to EOF checks each member's ZIP CRC.
            with zipped.open(info) as source, temporary.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
            temporary.replace(target)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="data")
    parser.add_argument("--offline", action="store_true",
                        help="Use local ZIPs or extracted PNG folders only; never download")
    args = parser.parse_args()
    root = Path(args.data)
    cache = root / "archives"
    cache.mkdir(parents=True, exist_ok=True)
    print("Official dataset: https://data.vision.ee.ethz.ch/cvl/DIV2K/")
    print("Downloads and extracted images need several GB of free disk space.")
    for name, folder, expected_count in ARCHIVES:
        marker = cache / (name + ".extracted")
        if marker.exists() and len(list((root / folder).glob("*.png"))) == expected_count:
            print(f"Already extracted: {folder}")
            continue
        archive = cache / name
        if args.offline and not archive.is_file():
            count = len(list((root / folder).glob("*.png")))
            if count == expected_count:
                print(f"Using existing PNG folder: {folder}; run verify_data.py for full checks.")
                continue
            parser.error(f"Offline mode: place {name} in {cache}, or provide "
                         f"{expected_count} extracted PNGs in {root / folder}. Found {count}.")
        download(BASE + name, archive)
        print(f"Extracting {name}...", flush=True)
        extract_pngs(archive, root)
        count = len(list((root / folder).glob("*.png")))
        if count != expected_count:
            raise ValueError(f"Expected {expected_count} PNGs in {folder}, found {count}.")
        marker.write_text("PNG extraction completed; ZIP CRC checked.\n", encoding="utf-8")
    print("DIV2K x4 pairs are ready.")


if __name__ == "__main__":
    main()
