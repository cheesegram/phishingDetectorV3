from __future__ import annotations

import argparse
import hashlib
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dhash_64(path: Path) -> int:
    with Image.open(path) as im:
        im = im.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = list(im.getdata())

    bits = 0
    bitpos = 0
    for y in range(8):
        row = y * 9
        for x in range(8):
            left = pixels[row + x]
            right = pixels[row + x + 1]
            if left > right:
                bits |= 1 << bitpos
            bitpos += 1
    return bits


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def find(parent: list[int], x: int) -> int:
    while parent[x] != x:
        parent[x] = parent[parent[x]]
        x = parent[x]
    return x


def union(parent: list[int], a: int, b: int) -> None:
    ra, rb = find(parent, a), find(parent, b)
    if ra != rb:
        parent[rb] = ra


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean exact and near-duplicate PNG images.")
    parser.add_argument(
        "--folder",
        default="datasets/phishing_website_image_datasets/circl_phishing_website_imageset/phishing",
        help="Folder containing PNG images to clean.",
    )
    parser.add_argument(
        "--near-threshold",
        type=int,
        default=5,
        help="Hamming distance threshold for near duplicates (dHash).",
    )
    parser.add_argument(
        "--review-folder",
        default="datasets/phishing_website_image_datasets/circl_phishing_website_imageset/_duplicate_review/phishing_near",
        help="Folder where near-duplicate images are moved.",
    )
    args = parser.parse_args()

    folder = Path(args.folder)
    review_folder = Path(args.review_folder)

    if not folder.exists() or not folder.is_dir():
        raise FileNotFoundError(f"Folder not found: {folder}")

    pngs = sorted(folder.glob("*.png"))
    print(f"Scanning folder: {folder}")
    print(f"PNG files before cleanup: {len(pngs)}")

    # 1) Remove exact duplicates by SHA-256 (keep lexicographically first file in each group).
    hash_groups: dict[str, list[Path]] = defaultdict(list)
    for p in pngs:
        hash_groups[sha256_of_file(p)].append(p)

    exact_groups = [sorted(group, key=lambda x: x.name) for group in hash_groups.values() if len(group) > 1]
    exact_removed = 0
    for group in exact_groups:
        keep = group[0]
        for duplicate in group[1:]:
            duplicate.unlink(missing_ok=True)
            exact_removed += 1
        print(f"Exact group size={len(group)} kept={keep.name}")

    # Refresh file list after exact duplicate removal.
    remaining_pngs = sorted(folder.glob("*.png"))

    # 2) Move near duplicates (keep lexicographically first in each cluster).
    hashes: list[tuple[Path, int]] = []
    for p in remaining_pngs:
        try:
            hashes.append((p, dhash_64(p)))
        except Exception:
            # Skip unreadable image files.
            pass

    parent = list(range(len(hashes)))
    for i in range(len(hashes)):
        for j in range(i + 1, len(hashes)):
            if hamming(hashes[i][1], hashes[j][1]) <= args.near_threshold:
                union(parent, i, j)

    clusters: dict[int, list[Path]] = defaultdict(list)
    for i, (path, _) in enumerate(hashes):
        clusters[find(parent, i)].append(path)

    near_clusters = [sorted(cluster, key=lambda x: x.name) for cluster in clusters.values() if len(cluster) > 1]
    near_moved = 0
    review_folder.mkdir(parents=True, exist_ok=True)

    for cluster in near_clusters:
        keep = cluster[0]
        for duplicate in cluster[1:]:
            target = review_folder / duplicate.name
            suffix = 1
            while target.exists():
                target = review_folder / f"{duplicate.stem}_{suffix}{duplicate.suffix}"
                suffix += 1
            shutil.move(str(duplicate), str(target))
            near_moved += 1
        print(f"Near cluster size={len(cluster)} kept={keep.name}")

    final_pngs = sorted(folder.glob("*.png"))

    print("\nCleanup summary")
    print(f"Exact duplicate groups: {len(exact_groups)}")
    print(f"Exact duplicates removed: {exact_removed}")
    print(f"Near-duplicate clusters: {len(near_clusters)}")
    print(f"Near duplicates moved: {near_moved}")
    print(f"PNG files after cleanup: {len(final_pngs)}")
    print(f"Near duplicates moved to: {review_folder}")


if __name__ == "__main__":
    main()
