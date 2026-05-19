# app.py

# =========================
# IMPORT THƯ VIỆN
# =========================

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# GIAO DIỆN WEB
# =========================

st.title("Dự báo giá cổ phiếu 10 ngày tới bằng LSTM")

st.write("Nhập mã cổ phiếu Việt Nam ví dụ: ACB.VN, FPT.VN, HPG.VN")

ticker = st.text_input("Nhập mã cổ phiếu", "ACB.VN")

# =========================
# NÚT DỰ BÁO
# =========================

if st.button("Dự báo"):

    # =========================
    # LẤY DỮ LIỆU
    # =========================

    df = yf.download(
        ticker,
        start="2018-01-01",
        end="2025-04-11"
    )

    # Kiểm tra dữ liệu

    if len(df) == 0:
        st.error("Không tìm thấy mã cổ phiếu")
        st.stop()

    # =========================
    # GIÁ CLOSE
    # =========================

    data = df[['Close']].values

    # =========================
    # CHIA TRAIN / TEST
    # =========================

    train_size = int(0.8 * len(data))

    train_data = data[:train_size]
    test_data = data[train_size:]

    # =========================
    # SCALE
    # =========================

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)

    test_scaled = scaler.transform(test_data)

    # =========================
    # TẠO SEQUENCE
    # =========================

    def create_sequences(data, window_size=60):

        X = []
        y = []

        for i in range(window_size, len(data)):

            X.append(data[i-window_size:i])

            y.append(data[i])

        return np.array(X), np.array(y)

    # Train

    X_train, y_train = create_sequences(train_scaled)

    # Reshape

    X_train = X_train.reshape(
        (X_train.shape[0],
         X_train.shape[1],
         1)
    )

    # =========================
    # XÂY DỰNG LSTM
    # =========================

    model = Sequential()

    model.add(
        LSTM(
            50,
            return_sequences=True,
            input_shape=(60,1)
        )
    )

    model.add(LSTM(50))

    model.add(Dense(1))

    model.compile(
        optimizer='adam',
        loss='mean_squared_error'
    )

    # =========================
    # TRAIN MODEL
    # =========================

    with st.spinner("Đang train mô hình LSTM..."):

        model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )

    st.success("Train mô hình thành công!")

    # =========================
    # DỰ BÁO 10 NGÀY TỚI
    # =========================

    last_60_days = np.vstack(
        (train_scaled, test_scaled)
    )[-60:]

    last_60_days = last_60_days.reshape(1,60,1)

    future_predictions = []

    for i in range(10):

        next_pred = model.predict(
            last_60_days,
            verbose=0
        )

        future_predictions.append(next_pred[0,0])

        last_60_days = np.append(
            last_60_days[:,1:,:],
            next_pred.reshape(1,1,1),
            axis=1
        )

    # =========================
    # INVERSE SCALE
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

    st.subheader("Giá dự báo 10 ngày tới")

    future_df = pd.DataFrame({
        "Ngày": range(1,11),
        "Giá dự báo": future_predictions.flatten()
    })

    st.dataframe(future_df)

    # =========================
    # VẼ BIỂU ĐỒ
    # =========================

    st.subheader("Biểu đồ dự báo")

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(
        range(1,11),
        future_predictions,
        marker='o'
    )

    ax.set_xlabel("Ngày")

    ax.set_ylabel("Giá cổ phiếu")

    ax.set_title(
        f"Dự báo giá cổ phiếu {ticker} trong 10 ngày tới"
    )

    ax.grid(True)

    st.pyplot(fig)
