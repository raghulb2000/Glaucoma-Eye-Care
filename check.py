import os
import hashlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TRAIN_DIR = os.path.join(BASE_DIR, "Dataset_split", "train")
VAL_DIR = os.path.join(BASE_DIR, "Dataset_split", "val")


def file_hash(path):
    hasher = hashlib.md5()

    with open(path, "rb") as f:
        while True:
            chunk = f.read(8192)

            if not chunk:
                break

            hasher.update(chunk)

    return hasher.hexdigest()


def collect_hashes(folder):
    hashes = {}

    for root, dirs, files in os.walk(folder):
        for file in files:

            if file.lower().endswith(
                (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
            ):
                path = os.path.join(root, file)

                h = file_hash(path)

                hashes.setdefault(h, []).append(path)

    return hashes


print("Checking dataset...")

train_hashes = collect_hashes(TRAIN_DIR)
val_hashes = collect_hashes(VAL_DIR)

duplicates = set(train_hashes) & set(val_hashes)

print("\nTrain images:", sum(len(v) for v in train_hashes.values()))
print("Validation images:", sum(len(v) for v in val_hashes.values()))

print("\nDuplicate images between train and validation:", len(duplicates))

if duplicates:
    print("\nWARNING: Data leakage detected!")

    for h in list(duplicates)[:10]:

        print("\nTRAIN:")
        for p in train_hashes[h]:
            print(p)

        print("VALIDATION:")
        for p in val_hashes[h]:
            print(p)

else:
    print("\nNo exact duplicate images found.")