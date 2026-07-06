# Quy trình xử lý dữ liệu SNN-Flood — Giải thích cho người mới

> Tài liệu học tập: giải thích toàn bộ quy trình xử lý dữ liệu "từ số 0", định nghĩa mọi thuật ngữ khi nó xuất hiện.
> **Liên quan:** [Quy trình xử lý (bản kỹ thuật)](SNN-Flood_QuyTrinh-XuLy-DuLieu.md) · [Pipeline tổng thể](SNN-Flood_Pipeline.md)

Cứ đọc từ trên xuống như một câu chuyện.

---

## Bức tranh lớn: chúng ta đang làm gì?

Máy tính không "nhìn" được ảnh vệ tinh như con người. Nó chỉ hiểu **những con số**. "Xử lý dữ liệu" nghĩa là biến đống ảnh thô + nhãn thô thành **những con số sạch, đúng định dạng** để mô hình học được. Giống như trước khi nấu ăn phải sơ chế: rửa, gọt, thái — chứ không quăng nguyên củ vào nồi.

Mục tiêu cuối: với mỗi ảnh radar của một vùng lũ, mô hình phải tô màu từng pixel thành *nền / nước thường trực / nước lũ*. Để dạy được điều đó, ta cần chuẩn bị **cặp (ảnh, đáp án)** thật chuẩn.

---

## Phần 1 — Hiểu dữ liệu đầu vào

**Ảnh SAR / Sentinel-1 là gì?** Sentinel-1 là vệ tinh của châu Âu mang một loại radar tên **SAR** (Synthetic Aperture Radar). Khác với máy ảnh thường (cần ánh sáng mặt trời), radar **tự phát sóng vi ba xuống đất rồi đo sóng dội lại** — giống con dơi định vị bằng tiếng kêu. Nhờ vậy nó "chụp" được cả **ban đêm và xuyên qua mây** — cực kỳ quan trọng vì lúc lũ trời thường mưa mù mịt, ảnh quang học (vệ tinh chụp ảnh thường) sẽ chỉ thấy mây.

**VV và VH là gì?** Sóng radar có "phương dao động" gọi là *phân cực*. Vệ tinh phát sóng dọc (V) rồi đo phần dội về theo phương dọc (→ kênh **VV**) và phương ngang (→ kênh **VH**). Nên mỗi ảnh có **2 lớp**, hiểu nôm na như ảnh màu có kênh Đỏ–Xanh vậy, nhưng đây là 2 "góc nhìn" radar khác nhau. Đó là lý do `s1` có shape `(2, 512, 512)`: 2 kênh, ảnh 512×512 pixel.

**Đơn vị dB và vì sao nước có màu tối.** Độ mạnh sóng dội được đo bằng **decibel (dB)** — một thang đo logarit (nén khoảng giá trị rất rộng lại cho dễ xử lý). Mặt nước phẳng phản xạ sóng đi *chỗ khác* (như gương hắt đèn pin ra xa mắt bạn), nên **rất ít sóng dội về vệ tinh → giá trị dB thấp → hiện màu tối**. Chính đặc điểm "nước thì tối" này là cái mô hình sẽ học để tìm ra vùng ngập.

**NaN là gì?** NaN = "Not a Number", tức **pixel không có dữ liệu** (thường ở rìa ảnh, nơi vệ tinh chưa quét tới). Ta phải đánh dấu và bỏ qua chúng, nếu không mô hình sẽ học nhầm từ chỗ trống.

---

## Phần 2 — Bài toán cốt lõi: "nước" ≠ "lũ" (Bước 1–4)

Đây là phần tinh tế nhất, và cũng là lý do đề tài phải chỉnh.

Nhãn gốc `LabelHand` chỉ nói mỗi pixel là **nước (1)** hay **không nước (0)**. Nhưng "nước" ở đây gộp cả **sông, hồ, ao vốn luôn có sẵn** lẫn **vùng mới bị ngập do lũ**. Mà con sông chảy quanh năm thì đâu phải "lũ"! Nếu mô hình chỉ tìm "nước", nó sẽ tính cả con sông vào diện tích ngập → sai bản chất bài toán "flood mapping".

**Cách tách ra — lớp JRC.** JRC (một viện nghiên cứu của EU) đã tổng hợp **hơn 30 năm ảnh vệ tinh** để lập bản đồ **nơi nào thường xuyên có nước**, gọi là *Global Surface Water*. Ta dùng nó như một "danh sách nước cố định". Logic rất đời thường:

> **Nước lũ = chỗ đang có nước NHƯNG không nằm trong danh sách nước cố định.**

Từ đó tạo **nhãn 3 lớp**:

| Giá trị | Ý nghĩa |
|:--:|---------|
| `0` | nền (đất khô) |
| `1` | nước thường trực (nằm trong bản đồ JRC) |
| `2` | **nước lũ** (đang có nước ∧ *không* thuộc JRC) ← thứ ta thực sự quan tâm |
| `-1` | pixel bỏ qua (NaN / không có dữ liệu) |

**Vì sao có nhãn `-1` "bỏ qua"?** Khi dạy mô hình, ta so đáp án của nó với nhãn đúng rồi "phạt" chỗ sai (gọi là *loss*). Nhưng ở pixel ta *không biết* sự thật (chỗ NaN), phạt hay thưởng đều vô nghĩa. Nên gắn `-1` để hàm loss **lơ nó đi** (`ignore_index=-1`). Giống chấm bài mà gặp câu bị nhòe mực thì bỏ qua, không trừ điểm.

Bước 1–4 chính là: ghép ảnh với nhãn, đọc chúng thành số, tạo mask nước cố định từ JRC, rồi ráp thành nhãn 3 lớp này.

---

## Phần 3 — Chia dữ liệu học/thi (Bước 5)

Trong ML, ta **không bao giờ** cho mô hình thi trên đúng bài nó đã học — vì nó có thể "học vẹt". Nên chia 3 phần:

- **Train (60%)** — sách giáo khoa để mô hình học.
- **Validation (20%)** — đề thi thử, để ta canh chỉnh trong lúc luyện.
- **Test (20%)** — đề thi thật, **chỉ dùng một lần cuối** để báo cáo điểm trung thực.

**"Phân tầng theo vùng" (stratified) nghĩa là gì?** Dữ liệu có 11 nước (USA, India, Paraguay...). Nếu chia ngẫu nhiên, xui thì toàn bộ ảnh Bolivia rơi hết vào test → mô hình chưa từng thấy Bolivia lúc học, điểm sẽ tệ oan. *Phân tầng* = chia đều **mỗi nước theo tỷ lệ 60/20/20**, để nước nào cũng góp mặt ở cả 3 tập. Kết quả thật: 265/90/90 chip, mỗi vùng đều có phần trong cả ba.

**"Rò rỉ không gian" (spatial leakage) — cái bẫy chết người.** Một vùng lũ được cắt thành nhiều ô vuông (tile) sát cạnh nhau, nên các ô kề nhau **trông rất giống nhau**. Nếu ô A vào tập train còn ô B (ngay cạnh) vào tập test, thì mô hình gần như đã "thấy trước đề thi" → điểm cao **ảo**. Đây là lỗi mà reviewer ngành ảnh vệ tinh soi rất kỹ. Cách phòng: chia theo **cụm vùng/cảnh** thay vì từng ô rời rạc.

**Vì sao cố định `seed=42`?** Máy tính xáo trộn bằng số ngẫu nhiên. "Seed" là hạt giống của bộ sinh ngẫu nhiên — cố định nó thì **lần nào chạy cũng ra đúng một cách chia**, để bạn (và người khác) *tái lập* được kết quả. Đây là nguyên tắc vàng của nghiên cứu: ai chạy lại cũng phải ra y hệt.

---

## Phần 4 — Làm sạch số cho mô hình (Bước 6–8)

**Chuẩn hóa (normalize) về [0,1].** Giá trị dB dao động khoảng −50 đến 0. Mô hình học *ổn định và nhanh hơn* khi mọi đầu vào nằm trong khoảng nhỏ gọn [0,1]. Công thức `(dB + 50) / 50` chỉ đơn giản là "kéo giãn" thang −50→0 thành 0→1. (Trước đó *clip* — cắt bỏ giá trị ngoài [−50,0] — để loại nhiễu rìa ảnh.)

**Crop 256×256.** Ảnh 512×512 nhân với nhiều bước thời gian (xem dưới) ngốn rất nhiều RAM/GPU. Lúc train ta cắt ngẫu nhiên một mảnh 256×256 cho nhẹ máy; lúc thi mới dùng ảnh đầy đủ.

---

## Phần 5 — "Dịch" ảnh sang ngôn ngữ của SNN (Bước 9)

Đề tài dùng **SNN (Spiking Neural Network)** — mạng nơ-ron *mô phỏng não thật*. Nơ-ron não không gửi số thực, chúng bắn ra **xung điện (spike)** — kiểu tín hiệu "có/không" theo thời gian. Ưu điểm: xung thưa thớt và chỉ là phép cộng đơn giản → **tốn rất ít điện** (đúng chữ "Energy-Efficient" trong tên đề tài).

Nhưng ảnh của ta là số thực tĩnh, nên phải **mã hóa (encode)** nó thành chuỗi xung kéo dài **T bước thời gian**:

- **Direct coding:** lặp lại ảnh qua T bước, để lớp nơ-ron đầu tiên tự sinh xung.
- **Rate coding:** pixel càng sáng thì xác suất bắn xung càng cao (đậm → bắn nhiều, nhạt → bắn ít).

`T` là số bước thời gian — càng lớn thì mô hình "nhìn" kỹ hơn nhưng tốn tính toán hơn. Đó là lý do ta sẽ làm *ablation* (thử nghiệm với T = 2, 4, 6, 8 để xem cái nào cân bằng nhất).

---

## Tóm tắt một câu mỗi bước

| Bước | Một câu dễ nhớ |
|:--:|----------------|
| 1 | Ghép ảnh với đáp án, vứt ảnh hỏng (toàn NaN). |
| 2 | Đọc ảnh & nhãn thành mảng số. |
| 3 | Dùng bản đồ JRC để biết đâu là nước "có sẵn". |
| 4 | Nước lũ = nước đang thấy trừ đi nước có sẵn → nhãn 3 lớp. |
| 5 | Chia học/thi-thử/thi-thật 60/20/20, đều các vùng, tránh học tủ. |
| 6 | Kéo giá trị về [0,1] cho mô hình học êm. |
| 7 | (Tùy chọn) thêm 1 kênh làm nổi bật nước. |
| 8 | Cắt ảnh nhỏ lại cho đỡ nặng máy. |
| 9 | Dịch ảnh thành chuỗi xung điện cho SNN "đọc". |

**Điểm mấu chốt cần nhớ:** cả pipeline xoay quanh việc tạo ra đáp án "nước lũ" thật chuẩn (nhờ JRC) và chia dữ liệu thật công bằng (tránh học tủ) — vì mô hình dù giỏi đến đâu cũng chỉ học tốt bằng chất lượng dữ liệu ta đưa vào.

---

*Tài liệu học tập — đọc kèm bản kỹ thuật để nắm cả "vì sao" lẫn "làm thế nào".*
