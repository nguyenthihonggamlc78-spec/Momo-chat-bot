import hashlib
import json
import os
import re
import shutil
import time
from datetime import datetime, timedelta, timezone

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# CẤU HÌNH
# ============================================================
st.set_page_config(page_title="MoMo - Trợ lý Mo-Mo", page_icon="💖", layout="centered")

MODEL = "gemini-3.6-flash"  # Đổi nếu mô hình không còn dùng được
FILE_DATA = "giao_dich_genz.csv"
FILE_DATA_GOC = "giao_dich_goc.csv"  # (không bắt buộc) bản sao dữ liệu mẫu
FILE_CHAT = "lich_su_chat.json"
FILE_TRANG_THAI = "trang_thai_ai.json"
HAN_MUC = {"Ăn uống": 1_500_000, "Mua sắm": 1_000_000, "Đu idol": 1_500_000}

# --- Giới hạn để tiết kiệm lượt gọi AI ---
LUOT_TOI_DA = 15  # Gói miễn phí hiện chỉ cho 20 lượt/ngày -> chừa dự phòng
MAX_LICH_SU = 6  # Chỉ gửi tối đa 6 tin gần nhất cho AI
PREFIX = "Ôi, Mo-Mo"  # Mọi tin báo lỗi/nghỉ đều bắt đầu bằng cụm này
DEBUG = True  # True: hiện chi tiết lỗi thật trong khung chat. Đổi False khi demo/nộp bài

SYSTEM_PROMPT = """
Bạn là Mo-Mo, trợ lý quản lý chi tiêu trong ứng dụng ví MoMo, nói chuyện với người dùng GenZ.

TÍNH CÁCH:
- Xưng "tớ", gọi người dùng là "cậu". Giọng bạn thân, hài hước, hơi mỏ hỗn, có thể trêu nhẹ.
- Dùng 1-2 emoji. Trả lời 3-5 câu, đủ ý theo thứ tự: (1) nêu số liệu cụ thể liên quan đến câu hỏi,
  (2) nhận xét hài hước, (3) gợi ý hành động.
- Luôn viết số tiền đầy đủ, ví dụ 2.250.000đ (không viết tắt, không bỏ chữ số).

QUY TẮC:
- CHỈ dùng số liệu trong phần [DỮ LIỆU] bên dưới. Tuyệt đối không tự bịa số.
- Nếu không có dữ liệu để trả lời, hãy nói thẳng là chưa có thông tin.
- Khi chi tiêu vượt hạn mức, cảnh báo rõ ràng rồi mới đùa.
- Luôn kết thúc bằng một gợi ý hành động cụ thể (cắt giảm, chuyển tiền tiết kiệm...).

GIỚI HẠN (GUARDRAILS):
- Không khuyên đầu tư cụ thể (cổ phiếu, coin, vay nóng).
- Không xúc phạm cá nhân, không đùa về hoàn cảnh khó khăn, nợ nần nghiêm trọng.
- Không trả lời câu hỏi ngoài chủ đề tài chính cá nhân; lịch sự đưa người dùng quay lại chủ đề.
"""

client = genai.Client(
    api_key=st.secrets["GEMINI_API_KEY"],
    http_options=types.HttpOptions(timeout=90000),
)


# ============================================================
# 1. ĐỌC VÀ TỔNG HỢP DỮ LIỆU
# ============================================================
def doc_du_lieu():
    return pd.read_csv(FILE_DATA, parse_dates=["ngay"], encoding="utf-8-sig")


def tom_tat(df):
    thu = df[df.loai == "thu"].so_tien.sum()
    chi = df[df.loai == "chi"]
    tong_chi = chi.so_tien.sum()
    theo_nhom = chi.groupby("danh_muc").so_tien.sum().sort_values(ascending=False)
    dong = [
        f"- {k}: {v:,.0f}đ ({v / tong_chi:.0%} tổng chi)" for k, v in theo_nhom.items()
    ]
    canh_bao = [
        f"- {k}: đã chi {theo_nhom[k]:,.0f}đ, vượt hạn mức {theo_nhom[k] - h:,.0f}đ"
        for k, h in HAN_MUC.items()
        if theo_nhom.get(k, 0) > h
    ]
    return (
        f"[DỮ LIỆU]\nTổng thu: {thu:,.0f}đ\nTổng chi: {tong_chi:,.0f}đ\n"
        f"Số dư: {thu - tong_chi:,.0f}đ\nChi theo nhóm:\n"
        + "\n".join(dong)
        + "\nCảnh báo vượt hạn mức:\n"
        + ("\n".join(canh_bao) or "- Không có")
    )


def nhan_xet_nhanh(df, nhom, tien):
    """Nhận xét tức thì bằng Python, KHÔNG tốn lượt gọi AI."""
    chi = df[df.loai == "chi"]
    tong_nhom = chi[chi.danh_muc == nhom].so_tien.sum()
    dong = f"✅ Đã ghi {tien:,.0f}đ vào nhóm {nhom}. Nhóm này đã chi {tong_nhom:,.0f}đ"
    han = HAN_MUC.get(nhom)
    if han:
        if tong_nhom > han:
            dong += f", vượt hạn mức {tong_nhom - han:,.0f}đ ⚠️"
        else:
            dong += f", còn {han - tong_nhom:,.0f}đ so với hạn mức."
    return dong


# ============================================================
# 2. QUẢN LÝ LƯỢT GỌI AI (đếm lượt, khóa tạm khi lỗi, bộ nhớ đệm)
# ============================================================
def hom_nay():
    # Quota của Google tính theo ngày giờ Thái Bình Dương (xấp xỉ UTC-8)
    return datetime.now(timezone(timedelta(hours=-8))).strftime("%Y-%m-%d")


def doc_trang_thai():
    tt = {"ngay": hom_nay(), "da_dung": 0, "khoa_den": 0.0}
    if os.path.exists(FILE_TRANG_THAI):
        try:
            with open(FILE_TRANG_THAI, encoding="utf-8") as f:
                du = json.load(f)
            if du.get("ngay") == tt["ngay"]:
                tt["da_dung"] = int(du.get("da_dung", 0))
            tt["khoa_den"] = float(du.get("khoa_den", 0.0))
        except Exception:
            pass
    return tt


def luu_trang_thai(tt):
    try:
        with open(FILE_TRANG_THAI, "w", encoding="utf-8") as f:
            json.dump(tt, f)
    except Exception:
        pass


def khoa_con_lai(tt=None):
    tt = tt or doc_trang_thai()
    return max(0.0, tt["khoa_den"] - time.time())


def dinh_dang_thoi_gian(giay):
    giay = int(giay)
    if giay >= 3600:
        return f"{giay // 3600} giờ {(giay % 3600) // 60} phút"
    if giay >= 60:
        return f"{giay // 60} phút"
    return f"{max(giay, 1)} giây"


def kiem_tra_cho_phep():
    """Kiểm tra TRƯỚC khi gọi API. Nếu không được phép thì không gửi yêu cầu nào."""
    tt = doc_trang_thai()
    con = khoa_con_lai(tt)
    if con > 0:
        return False, (
            f"{PREFIX} đang nghỉ ngơi 😴 "
            f"Cậu quay lại sau khoảng {dinh_dang_thoi_gian(con)} nhé."
        )
    if tt["da_dung"] >= LUOT_TOI_DA:
        return False, f"{PREFIX} đã dùng hết {LUOT_TOI_DA} lượt hỏi hôm nay 😴 Mai cậu quay lại nhé!"
    return True, ""


def ghi_nhan_luot():
    tt = doc_trang_thai()
    tt["da_dung"] += 1
    luu_trang_thai(tt)


def dat_khoa(giay):
    tt = doc_trang_thai()
    tt["khoa_den"] = time.time() + giay
    luu_trang_thai(tt)


def doc_thoi_gian_cho(loi):
    """Lấy số giây Google yêu cầu chờ từ thông báo lỗi 429."""
    m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", loi)
    if m:
        return float(m.group(1))
    m = re.search(r"retry in\s*(?:(\d+)h)?\s*(?:(\d+)m)?\s*(?:(\d+(?:\.\d+)?)s)?", loi)
    if m and any(m.groups()):
        h, mi, s = (float(x) if x else 0.0 for x in m.groups())
        return h * 3600 + mi * 60 + s
    return None


def xu_ly_loi(e):
    """Đổi lỗi thành câu thân thiện và khóa tạm để không gửi lại liên tục."""
    loi = str(e)
    print("LỖI GỌI AI:", loi[:800])  # xem đầy đủ trong Terminal / Manage app > Logs
    chi_tiet = f" [Chi tiết: {loi[:160]}]" if DEBUG else ""
    thap = loi.lower()
    if "429" in loi or "RESOURCE_EXHAUSTED" in loi:
        cho = doc_thoi_gian_cho(loi)
        if cho is None:
            cho = 6 * 3600 if ("PerDay" in loi or "per day" in loi) else 65
        dat_khoa(cho + 5)
        return f"{PREFIX} hết lượt hỏi miễn phí rồi 😴 Cậu quay lại sau khoảng {dinh_dang_thoi_gian(cho)} nhé."
    if "timed out" in thap or "timeout" in thap or "DEADLINE" in loi:
        dat_khoa(10)
        return f"{PREFIX} chờ AI trả lời quá lâu ⏳ Cậu thử lại nhé.{chi_tiet}"
    if "503" in loi or "UNAVAILABLE" in loi or ("500" in loi and "INTERNAL" in loi):
        dat_khoa(20)
        return f"{PREFIX} đang quá tải 😵 Cậu đợi khoảng 20 giây rồi hỏi lại nhé.{chi_tiet}"
    if "404" in loi or "NOT_FOUND" in loi:
        dat_khoa(60)
        return f"{PREFIX} không tìm thấy mô hình AI 😵 Cần kiểm tra lại tên mô hình (biến MODEL).{chi_tiet}"
    if "400" in loi or "INVALID_ARGUMENT" in loi:
        dat_khoa(10)
        return f"{PREFIX} gửi yêu cầu chưa đúng định dạng 😵 Cần kiểm tra lại cấu hình gọi AI.{chi_tiet}"
    dat_khoa(10)
    return f"{PREFIX} đang bị lỗi kết nối 😵 Cậu thử lại sau ít phút nhé.{chi_tiet}"


@st.cache_resource
def bo_nho_dem():
    return {}


def chuan_bi_lich_su(lich_su):
    """Bỏ tin báo lỗi và các câu hỏi đã thất bại, chỉ lấy tin gần nhất."""
    sach = []
    for i, m in enumerate(lich_su):
        if m["text"].startswith(PREFIX):
            continue
        sau = lich_su[i + 1] if i + 1 < len(lich_su) else None
        if m["role"] == "user" and sau and sau["text"].startswith(PREFIX):
            continue  # câu hỏi này đã thất bại, không gửi lại cho AI
        sach.append(m)
    sach = sach[-MAX_LICH_SU:]
    while sach and sach[0]["role"] != "user":
        sach.pop(0)
    gop = []
    for m in sach:
        if gop and gop[-1]["role"] == m["role"]:
            gop[-1] = {"role": m["role"], "text": gop[-1]["text"] + "\n" + m["text"]}
        else:
            gop.append(dict(m))
    return gop


# ============================================================
# 3. GỌI MÔ HÌNH AI (mỗi câu hỏi tối đa 1 lượt, không tự thử lại)
# ============================================================
def hoi_momo_stream(lich_su, df, max_tokens=1500):
    gui = chuan_bi_lich_su(lich_su)
    if not gui:
        yield f"{PREFIX} chưa nhận được câu hỏi nào 🤔"
        return

    tom_tat_du_lieu = tom_tat(df)
    khoa = hashlib.md5(
        (tom_tat_du_lieu + "||" + "||".join(m["text"] for m in gui[-3:])).encode("utf-8")
    ).hexdigest()

    # (a) Câu hỏi đã trả lời với cùng dữ liệu -> dùng lại, KHÔNG tốn lượt
    da_co = bo_nho_dem().get(khoa)
    if da_co:
        yield da_co
        return

    # (b) Đang bị khóa hoặc hết lượt trong ngày -> không gửi yêu cầu nào
    ok, thong_bao = kiem_tra_cho_phep()
    if not ok:
        yield thong_bao
        return

    # (c) Gọi API đúng 1 lần
    ghi_nhan_luot()
    noi_dung = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])]) for m in gui
    ]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + "\n\n" + tom_tat_du_lieu,
        temperature=0.8,
        max_output_tokens=max_tokens,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    cac_doan = []
    ly_do = ""
    try:
        for chunk in client.models.generate_content_stream(
            model=MODEL, contents=noi_dung, config=cfg
        ):
            try:
                if chunk.candidates[0].finish_reason:
                    ly_do = str(chunk.candidates[0].finish_reason)
            except Exception:
                pass
            if chunk.text:
                cac_doan.append(chunk.text)
                yield chunk.text
    except Exception as e:
        thong_bao_loi = xu_ly_loi(e)
        if not cac_doan:  # đã có chữ rồi thì giữ nguyên, chỉ khóa tạm
            yield thong_bao_loi
        return

    day_du = "".join(cac_doan).strip()
    if not day_du:
        yield f"{PREFIX} nghĩ mãi mà chưa ra câu trả lời 🤔 Cậu hỏi lại giúp tớ nhé."
        return
    if "MAX_TOKENS" in ly_do:  # câu bị cụt: báo cho người dùng, không lưu cache
        yield " …(tớ nói chưa hết, cậu hỏi tiếp để tớ nói thêm nhé)"
        return
    bo_nho_dem()[khoa] = day_du


# ============================================================
# 4. LƯU / TẢI LỊCH SỬ HỘI THOẠI
# ============================================================
def tai_lich_su():
    if os.path.exists(FILE_CHAT):
        try:
            with open(FILE_CHAT, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []
    return []


def luu_lich_su(ls):
    with open(FILE_CHAT, "w", encoding="utf-8") as f:
        json.dump(ls, f, ensure_ascii=False, indent=2)


# ===== GIAO DIỆN =====
if "lich_su" not in st.session_state:
    st.session_state.lich_su = tai_lich_su()

df = doc_du_lieu()

# ============================================================
# 5. THANH BÊN
# ============================================================
with st.sidebar.form("giao_dich_moi"):
    st.subheader("➕ Thêm giao dịch")
    mo_ta = st.text_input("Mô tả", "Trà sữa full topping")
    nhom = st.selectbox("Nhóm", ["Ăn uống", "Mua sắm", "Đu idol", "Khác"])
    tien = st.number_input("Số tiền (đ)", min_value=1000, step=1000, value=50000)
    gui = st.form_submit_button("Ghi lại")

if gui:
    moi = pd.DataFrame(
        [
            {
                "ngay": pd.Timestamp.today().strftime("%Y-%m-%d"),
                "mo_ta": mo_ta,
                "danh_muc": nhom,
                "so_tien": tien,
                "loai": "chi",
            }
        ]
    )
    moi.to_csv(FILE_DATA, mode="a", header=False, index=False, encoding="utf-8")
    df = doc_du_lieu()
    # Nhận xét bằng Python: tức thì và không tốn lượt AI
    st.session_state.lich_su.append(
        {"role": "user", "text": f"Tớ vừa ghi {tien:,.0f}đ cho '{mo_ta}' ({nhom})."}
    )
    st.session_state.lich_su.append(
        {"role": "model", "text": nhan_xet_nhanh(df, nhom, tien)}
    )
    luu_lich_su(st.session_state.lich_su)
    st.rerun()

if st.sidebar.button("🗑️ Xóa lịch sử chat"):
    st.session_state.lich_su = []
    luu_lich_su([])
    st.rerun()

if os.path.exists(FILE_DATA_GOC) and st.sidebar.button("🔄 Đặt lại dữ liệu mẫu"):
    shutil.copy(FILE_DATA_GOC, FILE_DATA)
    st.session_state.lich_su = []
    luu_lich_su([])
    st.rerun()

tt = doc_trang_thai()
con_khoa = khoa_con_lai(tt)
het_luot = tt["da_dung"] >= LUOT_TOI_DA
ai_nghi = con_khoa > 0 or het_luot
st.sidebar.caption(f"Lượt hỏi AI hôm nay: {tt['da_dung']}/{LUOT_TOI_DA}")
if con_khoa > 0:
    st.sidebar.warning(f"AI tạm nghỉ, thử lại sau khoảng {dinh_dang_thoi_gian(con_khoa)}.")
elif het_luot:
    st.sidebar.warning("Đã hết lượt hỏi AI hôm nay.")

# ============================================================
# 6. GIAO DIỆN MÔ PHỎNG APP MOMO
# ============================================================
st.markdown(
    """
<style>
.stApp { background:#fdf2f8; }
.momo-header { background:#A50064; color:white; padding:16px 20px;
               border-radius:14px; margin-bottom:12px; }
</style>
<div class="momo-header"><h3 style="margin:0">💖 MoMo · Quản lý chi tiêu</h3>
<small>Trợ lý Mo-Mo luôn canh ví giúp cậu</small></div>
""",
    unsafe_allow_html=True,
)

thu = df[df.loai == "thu"].so_tien.sum()
chi = df[df.loai == "chi"].so_tien.sum()
c1, c2, c3 = st.columns(3)
c1.metric("Số dư", f"{thu - chi:,.0f}đ")
c2.metric("Tổng thu", f"{thu:,.0f}đ")
c3.metric("Tổng chi", f"{chi:,.0f}đ")

st.caption("Chi tiêu theo nhóm")
st.bar_chart(df[df.loai == "chi"].groupby("danh_muc").so_tien.sum())

# ============================================================
# 7. KHUNG CHAT
# ============================================================
for m in st.session_state.lich_su:
    with st.chat_message("assistant" if m["role"] == "model" else "user"):
        st.write(m["text"])

cau_hoi = st.chat_input(
    "Mo-Mo đang nghỉ, quay lại sau nhé..." if ai_nghi else "Hỏi Mo-Mo về chi tiêu của cậu...",
    disabled=ai_nghi,
)
if cau_hoi:
    st.session_state.lich_su.append({"role": "user", "text": cau_hoi})
    with st.chat_message("user"):
        st.write(cau_hoi)
    with st.chat_message("assistant"):
        tra_loi = st.write_stream(hoi_momo_stream(st.session_state.lich_su, df))
    st.session_state.lich_su.append({"role": "model", "text": tra_loi})
    luu_lich_su(st.session_state.lich_su)
    st.rerun()  # cập nhật bộ đếm lượt ở thanh bên
