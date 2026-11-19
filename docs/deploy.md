# Deploy: API trên Hugging Face Space, giao diện trên Vercel

```
Trình duyệt → Vercel (deploy/web: trang + /api/extract) → Hugging Face Space (deploy/api: FastAPI + model, CPU) → JSON
```

- **API** (`deploy/api/app.py`): `GET /health`, `POST /extract` (ảnh, trường `file`). Cấu hình cuối + tổng tiền từ các
  món + chặn bịa chữ. Model nạp một lần khi Space khởi động; ảnh xử lý trong thư mục tạm, không lưu.
- **Web** (`deploy/web/`): trang `index.html` thu nhỏ ảnh còn ≤ 2000 px rồi gửi tới `/api/extract` của Vercel; hàm này
  giữ token và chuyển tiếp sang Space.

Cần: tài khoản Hugging Face, tài khoản Vercel (đăng nhập bằng GitHub). Bạn tự đăng nhập và tự tạo token.

## 1. Đưa trọng số lên Hugging Face Hub (một lần, trên máy)

1. huggingface.co → **New model** → tên `quethoadon-weights`, chọn **Private** → Create.
2. huggingface.co/settings/tokens → **Create new token**: một token **Write** (để upload) và một token **Read** (cho
   Space và Vercel). Không gửi token cho ai, không commit vào Git.
3. Trong PowerShell, ở thư mục dự án (thay `<user>` bằng tên tài khoản). `huggingface_hub` bản mới dùng lệnh `hf` thay cho `huggingface-cli`:

```powershell
pip install -U huggingface_hub
```

```powershell
hf auth login
```

```powershell
hf upload <user>/quethoadon-weights kaggle/m1b_v3_checkpoint/best m1b_v3_checkpoint/best
```

```powershell
hf upload <user>/quethoadon-weights kaggle/phase1_eval/vietocr_ft/vietocr_finetuned.pth vietocr_finetuned.pth
```

Repo phải có đúng: `m1b_v3_checkpoint/best/config.json`, `m1b_v3_checkpoint/best/model.safetensors`,
`vietocr_finetuned.pth` (khoảng 1,6 GB, upload vài phút đến vài chục phút).

## 2. Tạo Space chạy API

1. huggingface.co → **New Space** → tên `quethoadon-api`, **SDK: Docker**, template **Blank**, hardware **CPU basic
   (free)**, chọn **Private** (ảnh hóa đơn có tên, số điện thoại) → Create.
2. Tab **Files** → **Add file → Upload files**: tải lên `deploy/api/Dockerfile` và `deploy/api/README.md` (README có
   dòng `app_port: 7860`). Dockerfile tự lấy code từ nhánh `thinh-dev` trên GitHub. Nên tải hai file từ GitHub
   (mở file trên nhánh `thinh-dev` → **Download raw file**) thay vì lấy từ máy Windows: Git trên Windows có thể đổi
   xuống dòng sang CRLF, làm hỏng các dòng `RUN ... \` của Dockerfile.
3. Tab **Settings → Variables and secrets**:
   - Variable `WEIGHTS_REPO` = `<user>/quethoadon-weights`
   - Secret `HF_TOKEN` = token **Read** ở bước 1
   - Secret `API_KEY` = một chuỗi ngẫu nhiên tự đặt (chặn người lạ gọi API)
4. Space tự build (15–25 phút lần đầu, xem tab **Logs**). Khi chạy, model tải và nạp thêm vài phút.
5. Kiểm tra (Space private nên phải gửi token):

```powershell
curl.exe -H "Authorization: Bearer <HF_TOKEN>" https://<user>-quethoadon-api.hf.space/health
```

Phải ra `"status":"ready"`. Rồi thử một ảnh:

```powershell
curl.exe -H "Authorization: Bearer <HF_TOKEN>" -H "x-api-key: <API_KEY>" -F "file=@receipt.jpg" https://<user>-quethoadon-api.hf.space/extract
```

`"status":"error"` thì đọc `error` và tab Logs (thường do sai `WEIGHTS_REPO` hoặc thiếu `HF_TOKEN`).

## 3. Deploy giao diện lên Vercel

1. vercel.com → **Add New → Project** → chọn repo GitHub `QuetHoaDon`.
2. **Root Directory**: `deploy/web`; **Framework Preset**: Other; nhánh: `thinh-dev` (hoặc `main` sau khi merge).
3. **Environment Variables**:
   - `SPACE_URL` = `https://<user>-quethoadon-api.hf.space`
   - `HF_TOKEN` = token **Read**
   - `API_KEY` = cùng giá trị với secret của Space
4. **Deploy** → mở link `https://<project>.vercel.app`, tải ảnh lên.

## Lưu ý

- **Tốc độ:** CPU miễn phí, ước chừng 10–30 giây mỗi ảnh (chưa đo trên Space). `deploy/web/vercel.json` cho hàm
  `/api/extract` tối đa 60 giây; ảnh lâu hơn sẽ lỗi hết giờ, khi đó thử lại hoặc nâng phần cứng Space.
- **Space ngủ** khi lâu không dùng: lần đầu sau đó mất vài phút để build lại và nạp model. Trang web tự kiểm tra
  `/api/health` mỗi 10 giây và báo "máy chủ đang khởi động".
- **Đổi code:** push lên nhánh `thinh-dev` rồi vào Space → **Settings → Factory rebuild** (Dockerfile clone lại code).
- **Một ảnh mỗi lần:** API xử lý tuần tự; nhiều người dùng cùng lúc sẽ phải chờ.
- **Riêng tư:** không lưu ảnh; vẫn nên giữ Space private và `API_KEY`, vì ảnh hóa đơn có thông tin cá nhân.
