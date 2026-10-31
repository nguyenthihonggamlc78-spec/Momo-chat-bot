import json
import os

import pandas as pd
import streamlit as st
from google import genai
from google.genai import types

# ============================================================
# CẤU HÌNH
# ============================================================
st.set_page_config(page_title="MoMo - Trợ lý Mo-Mo", page_icon="💖", layout="centered")

MODEL = "gemini-3.8-flash"  # Kiểm tra tên mô hình hiện hành trong Google AI Studio
FILE_DATA = "giao_dich_genz.csv"
FILE_CHAT = "lich_su_chat.json"
HAN_MUC = {"Ăn uống": 1_500_000, "Mua sắm": 1_000_000, "Đu idol": 1_500_000}

SYSTEM_PROMPT = """
Bạn là Mo-Mo, trợ lý quản lý chi tiêu trong ứng dụng ví MoMo, nói chuyện với người dùng GenZ.

TÍNH CÁCH:
- Xưng "tớ", gọi người dùng là "cậu". Giọng bạn thân, hài hước, hơi mỏ hỗn, có thể trêu nhẹ.
- Dùng 1-2 emoji mỗi câu trả lời, câu trả lời ngắn gọn (tối đa 5 câu).

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

client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])


# ============================================================
# 1. ĐỌC VÀ TỔNG HỢP DỮ LIỆU
# ============================================================
def doc_du_lieu():
    return pd.read_csv(FILE_DATA, parse_dates=["ngay"])


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


# ============================================================
# 2. GỌI MÔ HÌNH AI
# ============================================================
def hoi_momo(lich_su, df):
    noi_dung = [
        types.Content(role=m["role"], parts=[types.Part(text=m["text"])])
        for m in lich_su
    ]
    cfg = types.GenerateContentConfig(
        system_instruction=SYSTEM_PROMPT + "\n\n" + tom_tat(df),
        temperature=0.8,
        max_output_tokens=1000,
        thinking_config=types.ThinkingConfig(thinking_level="LOW"),
    )
    try:
        return client.models.generate_content(
            model=MODEL, contents=noi_dung, config=cfg
        ).text
    except Exception as e:
        return f"Ôi, Mo-Mo đang bị lỗi kết nối rồi 😵 Cậu thử lại sau nhé. (Chi tiết: {e})"


# ============================================================
# 3. LƯU / TẢI LỊCH SỬ HỘI THOẠI
# ============================================================
def tai_lich_su():
    if os.path.exists(FILE_CHAT):
        with open(FILE_CHAT, encoding="utf-8") as f:
            return json.load(f)
    return []


def luu_lich_su(ls):
    with open(FILE_CHAT, "w", encoding="utf-8") as f:
        json.dump(ls, f, ensure_ascii=False, indent=2)


if "lich_su" not in st.session_state:
    st.session_state.lich_su = tai_lich_su()

# ============================================================
# 4. THANH BÊN: THÊM GIAO DỊCH + XÓA LỊCH SỬ
# ============================================================
df = doc_du_lieu()

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
    moi.to_csv(FILE_DATA, mode="a", header=False, index=False)
    df = doc_du_lieu()
    yeu_cau = f"Tớ vừa chi {tien:,.0f}đ cho '{mo_ta}' (nhóm {nhom}). Nhận xét nhanh giúp tớ."
    st.session_state.lich_su.append({"role": "user", "text": yeu_cau})
    with st.spinner("Mo-Mo đang soi giao dịch..."):
        tra_loi = hoi_momo(st.session_state.lich_su, df)
    st.session_state.lich_su.append({"role": "model", "text": tra_loi})
    luu_lich_su(st.session_state.lich_su)
    st.rerun()

if st.sidebar.button("🗑️ Xóa lịch sử chat"):
    st.session_state.lich_su = []
    luu_lich_su([])
    st.rerun()

# ============================================================
# 5. GIAO DIỆN MÔ PHỎNG APP MOMO
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
# 6. KHUNG CHAT
# ============================================================
for m in st.session_state.lich_su:
    with st.chat_message("assistant" if m["role"] == "model" else "user"):
        st.write(m["text"])

if cau_hoi := st.chat_input("Hỏi Mo-Mo về chi tiêu của cậu..."):
    st.session_state.lich_su.append({"role": "user", "text": cau_hoi})
    with st.chat_message("user"):
        st.write(cau_hoi)
    with st.chat_message("assistant"):
        with st.spinner("Mo-Mo đang soi ví cậu..."):
            tra_loi = hoi_momo(st.session_state.lich_su, df)
        st.write(tra_loi)
    st.session_state.lich_su.append({"role": "model", "text": tra_loi})
    luu_lich_su(st.session_state.lich_su)
