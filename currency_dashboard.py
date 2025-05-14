import streamlit as st
import pandas as pd
import requests
import plotly.express as px
from datetime import datetime
from sklearn.linear_model import LinearRegression
import numpy as np

# 🌐 Локализация
LANGUAGES = {
    "🇬🇧 English": {
        "title": "💱 Currency Dashboard",
        "base_currency": "Base Currency",
        "compare_with": "💱 Compare with:",
        "start_date": "📅 Start Date",
        "end_date": "📅 End Date",
        "min": "Min",
        "avg": "Average",
        "max": "Max",
        "stats": "📊 Currency Statistics",
        "chart": "📈 Exchange Rate Chart + Forecast",
        "download": "💾 Download CSV",
        "error": "❌ Failed to load data. Check your connection or date range."
    },
    "🇺🇦 Українська": {
        "title": "💱 Валютний дашборд",
        "base_currency": "Базова валюта",
        "compare_with": "💱 Порівняти з:",
        "start_date": "📅 Початкова дата",
        "end_date": "📅 Кінцева дата",
        "min": "Мінімум",
        "avg": "Середнє",
        "max": "Максимум",
        "stats": "📊 Статистика валют",
        "chart": "📈 Графік та прогноз",
        "download": "💾 Завантажити CSV",
        "error": "❌ Не вдалося завантажити дані. Перевірте з'єднання або діапазон дат."
    }
}

# Флаги валют
CURRENCY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "CNY": "🇨🇳",
    "JPY": "🇯🇵",
    "CHF": "🇨🇭",
    "CAD": "🇨🇦",
    "AUD": "🇦🇺"
}

st.set_page_config(page_title="Currency Dashboard", layout="wide")

# --- Сайдбар: язык и базовая валюта ---
lang_choice = st.sidebar.selectbox("🌍 Language / Мова", list(LANGUAGES.keys()))
t = LANGUAGES[lang_choice]

ALL_CURRENCIES = list(CURRENCY_FLAGS.keys())
base_currency = st.sidebar.selectbox(t["base_currency"], ALL_CURRENCIES, index=0)

# --- Выбор валют для сравнения
available_targets = [cur for cur in ALL_CURRENCIES if cur != base_currency]
def display_currency(cur): return f"{CURRENCY_FLAGS.get(cur, '')} {cur}"
selected_currencies = st.sidebar.multiselect(
    t["compare_with"], available_targets, default=available_targets[:2], format_func=display_currency
)

# --- Инфо-блок
with st.sidebar.expander("ℹ️ Info / Про проєкт"):
    st.markdown("""
    This dashboard uses [Frankfurter API](https://www.frankfurter.app/) to show daily exchange rates.

    **Instructions:**
    - Select base currency 💶
    - Pick a date range 📅
    - Choose currencies to compare ✅
    - View trends and 7-day forecast 📊
    - Export data 💾

    Made with ❤️ and [Streamlit](https://streamlit.io)
    """)

# --- Заголовок без бага
st.markdown("## 💶 Live Exchange Rates Dashboard")
st.markdown("Compare 📈 currencies over time and get a 7-day forecast 🔮.")
st.markdown("---")

# --- Даты по умолчанию
default_start = datetime(2025, 1, 1).date()
default_end = datetime(2025, 5, 13).date()
col1, col2 = st.columns(2)
start_date = col1.date_input(t["start_date"], default_start, min_value=datetime(2000, 1, 1).date(), max_value=default_end)
end_date = col2.date_input(t["end_date"], default_end, min_value=start_date, max_value=datetime.today().date())

# --- Получение данных
@st.cache_data(show_spinner=False)
def fetch_rates(start, end, base, symbols):
    url = f"https://api.frankfurter.app/{start}..{end}?from={base}&to={','.join(symbols)}"
    response = requests.get(url)
    if response.status_code != 200:
        raise Exception("API request failed.")
    data = response.json().get('rates', {})
    if not data:
        raise Exception("No data returned from API.")
    df = pd.DataFrame(data).T
    df.index = pd.to_datetime(df.index)
    return df.sort_index()

# --- Прогноз (7 дней вперёд)
def add_forecast(df, days=7):
    df_forecast = df.copy()
    for col in df.columns:
        y = df[col].values
        x = np.arange(len(y)).reshape(-1, 1)
        model = LinearRegression().fit(x, y)
        x_future = np.arange(len(y), len(y) + days).reshape(-1, 1)
        y_future = model.predict(x_future)
        forecast_dates = pd.date_range(df.index[-1] + pd.Timedelta(days=1), periods=days)
        df_future = pd.DataFrame(y_future, index=forecast_dates, columns=[col])
        df_forecast = pd.concat([df_forecast, df_future])
    return df_forecast

# --- Основная логика
if start_date > end_date:
    st.warning("⚠️ Start date must be before end date.")
elif not selected_currencies:
    st.warning("⚠️ Please select at least one currency to compare.")
else:
    try:
        df = fetch_rates(start_date, end_date, base_currency, selected_currencies)

        if df.empty:
            st.warning("⚠️ No data available for the selected dates.")
        else:
            # Метрики
            st.markdown("### 📊 " + t["stats"])
            for currency in selected_currencies:
                flag = CURRENCY_FLAGS.get(currency, "")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.success(f"{flag} {currency} ({t['min']}): {df[currency].min():.2f}")
                with col2:
                    st.info(f"{flag} {currency} ({t['avg']}): {df[currency].mean():.2f}")
                with col3:
                    st.warning(f"{flag} {currency} ({t['max']}): {df[currency].max():.2f}")

            # Прогноз и график
            st.markdown("### 📈 " + t["chart"])
            df_extended = add_forecast(df)
            fig = px.line(df_extended, x=df_extended.index, y=selected_currencies,
                          labels={'value': 'Rate', 'index': 'Date'},
                          title=f"Rates + Forecast (next 7 days) relative to {base_currency}")
            fig.update_layout(
                title_x=0.5,
                legend=dict(orientation="h", yanchor="bottom", y=-0.3, xanchor="center", x=0.5),
                margin=dict(l=40, r=40, t=40, b=40),
                height=500
            )
            st.plotly_chart(fig, use_container_width=True)

            # Скачивание CSV
            csv = df.to_csv().encode('utf-8')
            st.download_button(t["download"], data=csv, file_name="exchange_rates.csv", mime='text/csv')

    except Exception as e:
        st.error(t["error"])
        st.code(str(e))

# --- Футер
st.markdown("---")
st.markdown("© 2025 Created by Your Name. Powered by 💻 Python & Streamlit. Data from 📡 Frankfurter API.")
