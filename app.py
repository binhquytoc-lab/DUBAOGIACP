# =========================
# IMPORT THƯ VIỆN
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import torch
import torch.nn as nn

# =========================
# GIAO DIỆN STREAMLIT
# =========================

st.title("Dự báo giá cổ phiếu 10 ngày bằng PyTorch LSTM")

st.write("Ứng dụng AI dự báo giá cổ phiếu bằng LSTM + PyTorch")

# =========================
# NHẬP MÃ CỔ PHIẾU
# =========================

ticker = st.text_input("Nhập mã cổ phiếu:", "AAPL")

# =========================
# NÚT DỰ BÁO
# =========================

if st.button("Dự báo"):

    # =========================
    # TẢI DỮ LIỆU
    # =========================

    df = yf.download(
        ticker,
        start="2018-01-01",
        end="2025-01-01"
    )

    # =========================
    # KIỂM TRA DỮ LIỆU
    # =========================

    if df.empty:
        st.error("Không tìm thấy mã cổ phiếu")
        st.stop()

    # =========================
    # GIÁ ĐÓNG CỬA
    # =========================

    data = df[['Close']]

    # =========================
    # HIỂN THỊ DỮ LIỆU
    # =========================

    st.subheader("Dữ liệu cổ phiếu")

    st.write(data.tail())

    # =========================
    # BIỂU ĐỒ GIÁ
    # =========================

    st.subheader("Biểu đồ giá đóng cửa")

    fig = plt.figure(figsize=(12,6))

    plt.plot(data)

    plt.xlabel("Thời gian")

    plt.ylabel("Giá")

    st.pyplot(fig)

    # =========================
    # CHUẨN HÓA DỮ LIỆU
    # =========================

    scaler = MinMaxScaler(feature_range=(0,1))

    scaled_data = scaler.fit_transform(data)

    # =========================
    # TẠO DATASET
    # =========================

    sequence_length = 60

    X = []
    y = []

    for i in range(sequence_length, len(scaled_data)):
        X.append(scaled_data[i-sequence_length:i])
        y.append(scaled_data[i])

    X = np.array(X)
    y = np.array(y)

    # =========================
    # CHUYỂN SANG TENSOR
    # =========================

    X_train = torch.tensor(X, dtype=torch.float32)

    y_train = torch.tensor(y, dtype=torch.float32)

    # =========================
    # XÂY DỰNG MÔ HÌNH LSTM
    # =========================

    class LSTMModel(nn.Module):

        def __init__(self):
            super(LSTMModel, self).__init__()

            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=50,
                num_layers=2,
                batch_first=True
            )

            self.fc = nn.Linear(50,1)

        def forward(self, x):

            out, _ = self.lstm(x)

            out = out[:, -1, :]

            out = self.fc(out)

            return out

    # =========================
    # KHỞI TẠO MODEL
    # =========================

    model = LSTMModel()

    # =========================
    # LOSS & OPTIMIZER
    # =========================

    criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001
    )

    # =========================
    # TRAIN MODEL
    # =========================

    epochs = 10

    progress_bar = st.progress(0)

    for epoch in range(epochs):

        outputs = model(X_train)

        loss = criterion(outputs, y_train)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        progress_bar.progress((epoch + 1) / epochs)

    st.success("Train model thành công!")

    # =========================
    # DỰ BÁO 10 NGÀY TỚI
    # =========================

    last_60_days = scaled_data[-60:]

    future_predictions = []

    current_input = last_60_days.copy()

    for i in range(10):

        input_data = torch.tensor(
            current_input.reshape(1,60,1),
            dtype=torch.float32
        )

        with torch.no_grad():

            prediction = model(input_data)

        predicted_value = prediction.numpy()[0][0]

        future_predictions.append(predicted_value)

        current_input = np.append(
            current_input[1:],
            [[predicted_value]],
            axis=0
        )

    # =========================
    # CHUYỂN VỀ GIÁ THẬT
    # =========================

    future_predictions = np.array(
        future_predictions
    ).reshape(-1,1)

    future_predictions = scaler.inverse_transform(
        future_predictions
    )

    # =========================
    # HIỂN THỊ KẾT QUẢ
    # =========================

    st.subheader("Dự báo giá 10 ngày tới")

    prediction_df = pd.DataFrame(
        future_predictions,
        columns=['Giá dự báo']
    )

    st.write(prediction_df)

    # =========================
    # VẼ BIỂU ĐỒ DỰ BÁO
    # =========================

    st.subheader("Biểu đồ dự báo")

    fig2 = plt.figure(figsize=(12,6))

    plt.plot(
        range(len(data)),
        data.values,
        label='Giá lịch sử'
    )

    future_x = range(
        len(data),
        len(data) + 10
    )

    plt.plot(
        future_x,
        future_predictions,
        label='Dự báo 10 ngày'
    )

    plt.legend()

    st.pyplot(fig2)
