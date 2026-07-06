"""
Tải TOÀN BỘ dữ liệu hand-labeled (S1Hand + LabelHand + JRCWaterHand) từ Google Cloud,
cho ĐÚNG các chip liệt kê trong splits/*.csv (đã có sẵn trong repo).

=> Trên máy Vast/Colab chỉ cần: git clone repo -> python download_data.py
   KHÔNG phải upload ~1GB thủ công. (Máy có mạng nhanh tải ~1-2 phút.)

Sau đó:  python build_3class_labels.py     # dựng Label3Class/ (splits đã có sẵn)
Chỉ dùng thư viện chuẩn (urllib) — không cần cài gì.
"""
import os
import csv
import glob
import time
import urllib.request
import urllib.error

GCS = ("https://storage.googleapis.com/sen1floods11/v1.1/data/"
       "flood_events/HandLabeled/")
LAYERS = {                       # thư mục : hậu tố tên file
    "S1Hand": "_S1Hand",
    "LabelHand": "_LabelHand",
    "JRCWaterHand": "_JRCWaterHand",
}


def chip_ids_from_splits(root="."):
    ids = set()
    for path in glob.glob(os.path.join(root, "splits", "*.csv")):
        with open(path) as f:
            for row in csv.DictReader(f):
                ids.add(row["chip_id"])
    return sorted(ids)


def download_layer(root, layer, suffix, ids, retries=3):
    out = os.path.join(root, layer)
    os.makedirs(out, exist_ok=True)
    ok = skip = fail = 0
    missing = []
    for i, cid in enumerate(ids, 1):
        dst = os.path.join(out, f"{cid}{suffix}.tif")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1
            continue
        url = f"{GCS}{layer}/{cid}{suffix}.tif"
        for attempt in range(retries):
            try:
                urllib.request.urlretrieve(url, dst)
                ok += 1
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    missing.append(cid)
                    fail += 1
                    if os.path.exists(dst):
                        os.remove(dst)
                    break
                time.sleep(1.5 * (attempt + 1))
            except Exception:
                time.sleep(1.5 * (attempt + 1))
        else:
            fail += 1
            missing.append(cid)
        if i % 100 == 0:
            print(f"  {layer}: {i}/{len(ids)} (tải {ok}, có sẵn {skip}, lỗi {fail})")
    print(f"{layer:14s}: tải {ok}, có sẵn {skip}, lỗi {fail}")
    return missing


def main(root="."):
    ids = chip_ids_from_splits(root)
    if not ids:
        raise SystemExit("Không thấy splits/*.csv — bạn đã clone đủ repo chưa?")
    print(f"Cần tải {len(ids)} chip × {len(LAYERS)} lớp từ Google Cloud\n")
    all_missing = {}
    for layer, suf in LAYERS.items():
        miss = download_layer(root, layer, suf, ids)
        if miss:
            all_missing[layer] = miss
    if all_missing:
        print("\n[Cảnh báo] Một số chip thiếu trên GCS:")
        for layer, miss in all_missing.items():
            print(f"  {layer}: {len(miss)} thiếu (vd {miss[:5]})")
    else:
        print("\n✅ Đủ toàn bộ 3 lớp. Bước tiếp: python build_3class_labels.py")


if __name__ == "__main__":
    main(".")
