"""
download_jrc.py — Tải lớp JRCWaterHand (nước thường trực) cho ĐÚNG 446 chip
hand-labeled mà bạn đang có, để dựng nhãn 3 lớp (build_3class_labels.py).

CHẠY TRÊN MÁY CỦA BẠN (không chạy trong sandbox — sandbox chặn mạng).
    cd dataset
    python download_jrc.py                 # nguồn chính: Google Cloud (public)
    python download_jrc.py --source hf     # dự phòng: mirror Hugging Face

Chỉ dùng thư viện chuẩn cho nguồn GCS (không cần cài gì).
Nguồn HF cần: pip install huggingface_hub
Kết quả: dataset/JRCWaterHand/{chip_id}_JRCWaterHand.tif
Sau đó:  python build_3class_labels.py
"""
import os, glob, argparse, time, urllib.request, urllib.error

S1_DIR, S1_SUF = "S1Hand", "_S1Hand"
JRC_DIR, JRC_SUF = "JRCWaterHand", "_JRCWaterHand"

# Bucket công khai chính thức của Sen1Floods11 (Cloud to Street)
GCS_BASE = ("https://storage.googleapis.com/sen1floods11/v1.1/data/"
            "flood_events/HandLabeled/JRCWaterHand/")

# Mirror Hugging Face (dùng khi GCS đổi đường dẫn). Đổi repo nếu cần.
HF_REPO = "harshinde/sen1floods11"


def chip_ids(root):
    """Lấy id từ tên ảnh S1Hand: Bolivia_103757_S1Hand.tif -> Bolivia_103757"""
    files = sorted(glob.glob(os.path.join(root, S1_DIR, f"*{S1_SUF}.tif")))
    return [os.path.basename(f).replace(f"{S1_SUF}.tif", "") for f in files]


def download_gcs(root, ids, retries=3):
    out = os.path.join(root, JRC_DIR)
    os.makedirs(out, exist_ok=True)
    ok = skip = fail = 0
    missing = []
    for i, cid in enumerate(ids, 1):
        dst = os.path.join(out, f"{cid}{JRC_SUF}.tif")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            skip += 1; continue
        url = f"{GCS_BASE}{cid}{JRC_SUF}.tif"
        for attempt in range(1, retries + 1):
            try:
                urllib.request.urlretrieve(url, dst)
                ok += 1
                break
            except urllib.error.HTTPError as e:
                if e.code == 404:
                    missing.append(cid); fail += 1
                    if os.path.exists(dst): os.remove(dst)
                    break
                time.sleep(1.5 * attempt)
            except Exception:
                time.sleep(1.5 * attempt)
        else:
            fail += 1; missing.append(cid)
        if i % 50 == 0:
            print(f"  ...{i}/{len(ids)}  (tải {ok}, bỏ qua {skip}, lỗi {fail})")
    print(f"\nGCS xong: tải mới {ok}, đã có {skip}, thất bại {fail}")
    if missing:
        print(f"  Thiếu {len(missing)} chip (ví dụ): {missing[:8]}")
        print("  -> thử lại bằng nguồn HF:  python download_jrc.py --source hf")
    return fail == 0


def download_hf(root, ids):
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Cần cài: pip install huggingface_hub"); return False
    out = os.path.join(root, JRC_DIR)
    os.makedirs(out, exist_ok=True)
    ok = fail = 0; missing = []
    # Thử vài tiền tố thư mục thường gặp trong các mirror
    prefixes = [
        "v1.1/data/flood_events/HandLabeled/JRCWaterHand/",
        "data/flood_events/HandLabeled/JRCWaterHand/",
        "HandLabeled/JRCWaterHand/",
        "JRCWaterHand/",
    ]
    for i, cid in enumerate(ids, 1):
        dst = os.path.join(out, f"{cid}{JRC_SUF}.tif")
        if os.path.exists(dst) and os.path.getsize(dst) > 0:
            continue
        got = False
        for pre in prefixes:
            try:
                p = hf_hub_download(HF_REPO, filename=f"{pre}{cid}{JRC_SUF}.tif",
                                    repo_type="dataset")
                import shutil; shutil.copy(p, dst); ok += 1; got = True; break
            except Exception:
                continue
        if not got:
            fail += 1; missing.append(cid)
        if i % 50 == 0:
            print(f"  ...{i}/{len(ids)}  (tải {ok}, lỗi {fail})")
    print(f"\nHF xong: tải {ok}, thất bại {fail}")
    if missing:
        print(f"  Vẫn thiếu {len(missing)} chip — kiểm tra lại repo/tiền tố HF, hoặc dùng phương án GEE (xem cuối file).")
    return fail == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--source", choices=["gcs", "hf"], default="gcs")
    args = ap.parse_args()

    ids = chip_ids(args.root)
    print(f"Số chip cần lớp JRC: {len(ids)}  (nguồn: {args.source.upper()})")
    if not ids:
        raise SystemExit("Không thấy S1Hand/*.tif — kiểm tra --root")

    done = download_hf(args.root, ids) if args.source == "hf" else download_gcs(args.root, ids)
    if done:
        print("\nĐủ 446 lớp JRC. Bước tiếp:  python build_3class_labels.py")

# --------------------------------------------------------------------------
# PHƯƠNG ÁN DỰ PHÒNG (nếu cả GCS và HF đều không có JRCWaterHand)
#
# A. gsutil (nhanh nhất, cần Google Cloud SDK, bucket công khai — không cần login):
#    gsutil -m cp \
#      "gs://sen1floods11/v1.1/data/flood_events/HandLabeled/JRCWaterHand/*" \
#      ./JRCWaterHand/
#
# B. Google Earth Engine — lấy JRC thô rồi tự cắt theo lưới từng chip S1Hand:
#    ee.Image("JRC/GSW1_4/GlobalSurfaceWater").select("occurrence")
#    reproject + clip theo transform/CRS của mỗi {id}_S1Hand.tif, ghi ra
#    {id}_JRCWaterHand.tif. build_3class_labels.py tự nhận occurrence 0-100
#    (permanent = occurrence >= --jrc_threshold, mặc định 50).
# --------------------------------------------------------------------------
