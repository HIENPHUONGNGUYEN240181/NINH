# python.py

import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo Cáo Tài chính 📊")

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    # Dùng .replace(0, 1e-9) cho Series Pandas để tránh lỗi chia cho 0
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    # Lọc chỉ tiêu "TỔNG CỘNG TÀI SẢN"
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        # Nếu không tìm thấy, cố gắng tìm Tổng Tài Sản
        try:
            tong_tai_san_N_1 = df[df['Chỉ tiêu'].str.contains('TÀI SẢN', case=False, na=False)]['Năm trước'].iloc[-1]
            tong_tai_san_N = df[df['Chỉ tiêu'].str.contains('TÀI SẢN', case=False, na=False)]['Năm sau'].iloc[-1]
        except IndexError:
             raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN' hoặc 'TÀI SẢN' nào.")
    else:
        tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
        tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini cho Phân tích Tài chính (Chức năng 5) ---
def get_ai_analysis(data_for_ai, api_key):
    """Gửi dữ liệu phân tích đến Gemini API và nhận nhận xét."""
    try:
        client = genai.Client(api_key=api_key)
        model_name = 'gemini-2.5-flash' 

        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Dựa trên các chỉ số tài chính sau, hãy đưa ra một nhận xét khách quan, ngắn gọn (khoảng 3-4 đoạn) về tình hình tài chính của doanh nghiệp. Đánh giá tập trung vào tốc độ tăng trưởng, thay đổi cơ cấu tài sản và khả năng thanh toán hiện hành.
        
        Dữ liệu thô và chỉ số:
        {data_for_ai}
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except KeyError:
        return "Lỗi: Không tìm thấy Khóa API 'GEMINI_API_KEY'. Vui lòng kiểm tra cấu hình Secrets trên Streamlit Cloud."
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định cho chỉ số thanh toán
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"
            
            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                # Cần đảm bảo Nợ Ngắn Hạn không bằng 0 để tránh lỗi chia cho 0
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]
                
                # Xử lý chia cho 0
                divisor_no_N = no_ngan_han_N if no_ngan_han_N != 0 else 1e-9
                divisor_no_N_1 = no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else 1e-9

                # Tính toán
                thanh_toan_hien_hanh_N = tsnh_n / divisor_no_N
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / divisor_no_N_1
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần"
                    )
                with col2:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                        value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                        delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                    )
                    
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số Thanh toán Hiện hành.")
                # Giữ N/A để tránh lỗi ở Chức năng 5
            except ZeroDivisionError:
                st.warning("Giá trị Nợ Ngắn Hạn bằng 0, không thể tính Chỉ số Thanh toán Hiện hành.")
                
            # --- Chức năng 5: Nhận xét AI ---
            st.subheader("5. Nhận xét Tình hình Tài chính (AI)")
            
            # Chuẩn bị dữ liệu để gửi cho AI
            # Cần đảm bảo thanh_toan_hien_hanh_N/N_1 là string để tránh lỗi format
            tt_n = f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else str(thanh_toan_hien_hanh_N)
            tt_n_1 = f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else str(thanh_toan_hien_hanh_N_1)
            
            data_for_ai = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    tt_n_1, 
                    tt_n
                ]
            }).to_markdown(index=False) 

            if st.button("Yêu cầu AI Phân tích"):
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if api_key:
                    with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                        ai_result = get_ai_analysis(data_for_ai, api_key)
                        st.markdown("**Kết quả Phân tích từ Gemini AI:**")
                        st.info(ai_result)
                else:
                    st.error("Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets.")

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    st.info("Vui lòng tải lên file Excel để bắt đầu phân tích.")

# ************************************************************************************************
# ------------------------------------ PHẦN THÊM KHUNG CHAT GEMINI --------------------------------
# ************************************************************************************************

st.divider()

st.subheader("Trợ lý Chatbot Phân tích Tài chính (Powered by Gemini) 🤖")
st.write("Bạn có thể hỏi các câu hỏi chung về phân tích tài chính, kế toán, hoặc yêu cầu giải thích các chỉ số.")

# 1. Lấy API Key và Khởi tạo Client
# Lấy API Key từ Streamlit Secrets, giống như cách đã dùng ở Chức năng 5
GEMINI_API_KEY = st.secrets.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.warning("⚠️ Không tìm thấy Khóa API 'GEMINI_API_KEY' trong Streamlit Secrets. Chatbot sẽ không hoạt động.")
    chat_client = None
else:
    try:
        chat_client = genai.Client(api_key=GEMINI_API_KEY)
    except Exception as e:
        st.error(f"Lỗi khởi tạo Gemini Client: {e}")
        chat_client = None

if chat_client:
    
    # 2. Khởi tạo Session State (để lưu lịch sử)
    if "chat_messages" not in st.session_state:
        st.session_state["chat_messages"] = []
    
    # Khởi tạo phiên chat (Gemini Chat Session) để giữ ngữ cảnh hội thoại
    if "chat_session" not in st.session_state:
        # Cấu hình hệ thống prompt để định danh vai trò của Trợ lý
        system_instruction = "Bạn là Trợ lý Chatbot Phân tích Tài chính thông minh. Bạn có thể trả lời các câu hỏi liên quan đến tài chính, kế toán, và phân tích kinh doanh một cách chuyên nghiệp và hữu ích. Giữ câu trả lời ngắn gọn và đi thẳng vào vấn đề."
        
        st.session_state["chat_session"] = chat_client.chats.create(
            model="gemini-2.5-flash", # Model tối ưu cho chat
            config=types.GenerateContentConfig(
                system_instruction=system_instruction
            )
        )

    # 3. Hiển thị Lịch sử Chat
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 4. Xử lý Input mới
    if prompt := st.chat_input("Đặt câu hỏi của bạn về tài chính..."):
        
        # Thêm tin nhắn người dùng vào lịch sử và hiển thị
        st.session_state["chat_messages"].append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Gửi tin nhắn và nhận phản hồi từ Gemini
        with st.chat_message("assistant"):
            with st.spinner("Đang chờ Gemini phản hồi..."):
                try:
                    # Gửi tin nhắn bằng phiên chat đã lưu
                    response = st.session_state["chat_session"].send_message(prompt)
                    
                    # Hiển thị và lưu phản hồi
                    st.markdown(response.text)
                    st.session_state["chat_messages"].append({"role": "assistant", "content": response.text})

                except Exception as e:
                    error_msg = f"Lỗi khi giao tiếp với Gemini: Vui lòng kiểm tra kết nối hoặc API Key. Chi tiết lỗi: {e}"
                    st.error(error_msg)
                    st.session_state["chat_messages"].append({"role": "assistant", "content": error_msg})
