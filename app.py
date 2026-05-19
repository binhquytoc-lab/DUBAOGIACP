import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# =========================
# GIAO DIỆN
# =========================

st.title("Dự báo giá cổ phiếu bằng LSTM")

ticker = st.text_input(
    "Nhập mã cổ phiếu",
    "ACB.VN"
)

# =========================
# BUTTON
# =========================

if st.button("Dự báo"):

    # =========================
    # DOWNLOAD DATA
    # =========================

    df = yf.download(
        ticker,
        start="2018-01-01",
        end="2025-04-11",
        auto_adjust=True
    )

    # =========================
    # KIỂM TRA DATA
    # =========================

    if df.empty:
        st.error("Không tìm thấy dữ liệu")
        st.stop()

    st.write(df.tail())

    # =========================
    # LẤY CLOSE PRICE
    # =========================

    data = df['Close'].values.reshape(-1,1)

    # =========================
    # TRAIN TEST
    # =========================

    train_size = int(len(data) * 0.8)

    train_data = data[:train_size]

    test_data = data[train_size:]

    # =========================
    # SCALE
    # =========================

    scaler = MinMaxScaler()

    train_scaled = scaler.fit_transform(train_data)

    test_scaled = scaler.transform(test_data)

    # =========================
    # CREATE SEQUENCE
    # =========================

    def create_sequences(data, window_size=60):

        X = []
        y = []

        for i in range(window_size, len(data)):

            X.append(data[i-window_size:i])

            y.append(data[i])

        return np.array(X), np.array(y)

    X_train, y_train = create_sequences(train_scaled)

    # =========================
    # RESHAPE
    # =========================

    X_train = X_train.reshape(
        X_train.shape[0],
        X_train.shape[1],
        1
    )

    # =========================
    # MODEL
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
    # TRAIN
    # =========================

    with st.spinner("Đang train model..."):

        model.fit(
            X_train,
            y_train,
            epochs=5,
            batch_size=32,
            verbose=0
        )

    st.success("Train thành công!")

    # =========================
    # FUTURE PREDICTION
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

        future_predictions.append(
            next_pred[0,0]
        )

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
    # DATAFRAME
    # =========================

    future_df = pd.DataFrame({

        "Ngày": range(1,11),

        "Giá dự báo":
        future_predictions.flatten()

    })

    st.subheader("Dự báo 10 ngày tới")

    st.dataframe(future_df)

    # =========================
    # PLOT
    # =========================

    fig, ax = plt.subplots(figsize=(12,6))

    ax.plot(

        range(1,11),

        future_predictions.flatten(),

        marker='o'

    )

    ax.set_title(
        f"Dự báo giá {ticker}"
    )

    ax.set_xlabel("Ngày")

    ax.set_ylabel("Giá")

    ax.grid(True)

    st.pyplot(fig)
