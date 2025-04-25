import streamlit as st
from datetime import datetime
from dateutil.relativedelta import relativedelta
import pandas as pd
from scipy.optimize import newton
import plotly.graph_objects as go

st.set_page_config(page_title="Bond Dashboard", layout="wide")

# ---------------------- COMPACT CSS ----------------------
st.markdown("""
    <style>
        .block-container {
            padding-top: 0.7rem;
            padding-bottom: 0.2rem;
            max-width: 95vw;
        }
        .stDataFrame { height: 160px !important; }

        .stNumberInput > div > div > input,
        .stDateInput > div > div > input,
        .stSelectbox > div > div > div {
            padding: 0.2rem 0.4rem;
            font-size: 0.75rem;
            height: 2rem;
        }
        label {
            font-size: 0.75rem !important;
            margin-bottom: 0.1rem !important;
        }
        .stNumberInput > div > div,
        .stDateInput > div > div,
        .stSelectbox > div > div {
            max-width: 180px;
        }
        section[data-testid="column"] > div {
            margin-bottom: -0.4rem;
        }
    </style>
""", unsafe_allow_html=True)

st.title("Bond Investment Dashboard")

# ---------------------- INPUT SECTION ----------------------
with st.container():
    input_col, gauge_col = st.columns([4, 1])

    with input_col:
        r1c1, r1c2, r1c3, r1c4 = st.columns(4)
        with r1c1:
            face_value = st.number_input("Face Value", value=10000.0)
        with r1c2:
            price = st.number_input("Purchase Price", value=10257.0)
        with r1c3:
            coupon_rate = st.number_input("Coupon Rate (%)", value=8.0)
        with r1c4:
            settlement_date = st.date_input("Settlement Date", value=datetime.today())

        r2c1, r2c2, r2c3, r2c4 = st.columns(4)
        with r2c1:
            frequency = st.selectbox("Coupon Frequency", ["Annual", "Semi-Annual", "Quarterly", "Monthly"])
        with r2c2:
            maturity_date = st.date_input("Maturity Date", value=datetime(2030, 12, 31))
        with r2c3:
            accrued_interest = st.number_input("Accrued Interest (₹)", value=0.0)
        with r2c4:
            st.markdown("")

# ---------------------- BOND CALCULATIONS ----------------------
settlement_date = pd.to_datetime(settlement_date)
maturity_date = pd.to_datetime(maturity_date)
freq_map = {"Annual": 1, "Semi-Annual": 2, "Quarterly": 4, "Monthly": 12}
interval_map = {1: relativedelta(years=1), 2: relativedelta(months=6),
                4: relativedelta(months=3), 12: relativedelta(months=1)}
freq = freq_map[frequency]
interval = interval_map[freq]
coupon_payment = face_value * (coupon_rate / 100) / freq

cf_dates = []
cur = settlement_date + interval
while cur < maturity_date:
    cf_dates.append(cur)
    cur += interval
cf_dates.append(maturity_date)

if len(cf_dates) > 1 and (cf_dates[-2] + interval) == cf_dates[-1]:
    cashflows = [coupon_payment] * (len(cf_dates) - 2) + [coupon_payment, face_value]
    labels = ["Coupon"] * (len(cf_dates) - 2) + ["Coupon", "Principal"]
else:
    cashflows = [coupon_payment] * (len(cf_dates) - 1) + [coupon_payment + face_value]
    labels = ["Coupon"] * (len(cf_dates) - 1) + ["Coupon + Principal"]

cf_df = pd.DataFrame({"Date": cf_dates, "Cashflow": cashflows, "Label": labels})
cf_df["Cumulative"] = cf_df["Cashflow"].cumsum()
cf_df["Year"] = cf_df["Date"].dt.year
cf_df["Period"] = cf_df["Date"].dt.to_period("M")

def bond_price(ytm):
    return sum([
        coupon_payment / (1 + ytm / freq) ** t for t in range(1, len(cf_df))
    ]) + (face_value + (coupon_payment if cf_df["Label"].iloc[-1] != "Principal" else 0)) / (1 + ytm / freq) ** len(cf_df)

def price_diff(ytm): return bond_price(ytm) - (price + accrued_interest)

try:
    ytm = newton(price_diff, 0.05) * 100
except:
    ytm = None

# ---------------------- GAUGE CHART ----------------------
fig_gauge = go.Figure(go.Indicator(
    mode="gauge+number+delta",
    value=ytm,
    delta={'reference': coupon_rate, 'increasing': {'color': "green"}, 'decreasing': {'color': "red"}},
    gauge={
        'axis': {'range': [0, 20]},
        'bar': {'color': "darkblue"},
        'steps': [
            {'range': [0, 5], 'color': "#e0e0e0"},
            {'range': [5, 10], 'color': "#fce5cd"},
            {'range': [10, 20], 'color': "#d9ead3"},
        ],
        'threshold': {
            'line': {'color': "blue", 'width': 4},
            'thickness': 0.75,
            'value': ytm
        }
    },
    title={'text': "YTM (%)"}
))
fig_gauge.update_layout(height=150, margin=dict(l=10, r=10, t=10, b=10))
with gauge_col:
    st.markdown("#### Yield to Maturity")
    st.plotly_chart(fig_gauge, use_container_width=True)

# ---------------------- INSIGHT ----------------------
if ytm:
    if price < face_value:
        st.success(f"🟢 **Discount** — Higher YTM of **{ytm:.2f}%** due to lower entry price. Accrued Interest: ₹{accrued_interest:,.2f}")
    elif price > face_value:
        st.warning(f"🔴 **Premium** — Lower YTM of **{ytm:.2f}%** due to higher entry price. Accrued Interest: ₹{accrued_interest:,.2f}")
    else:
        st.info(f"🟡 **Par** — YTM matches coupon rate. Accrued Interest: ₹{accrued_interest:,.2f}")

# ---------------------- CHARTS + TABLE ----------------------
with st.container():
    col1, col2, col3 = st.columns(3)

    cf_df["GroupLabel"] = cf_df["Year"] if freq == 1 else cf_df["Period"].astype(str)
    periodwise = cf_df.groupby("GroupLabel")["Cashflow"].sum().reset_index()
    initial_label = settlement_date.year if freq == 1 else settlement_date.strftime("%b-%Y")
    periodwise.loc[len(periodwise)] = [initial_label, -price]
    periodwise = periodwise.sort_values("GroupLabel")
    periodwise["Color"] = ["crimson" if x < 0 else ("seagreen" if i == len(periodwise)-1 else "steelblue") for i, x in enumerate(periodwise["Cashflow"])]

    # Cash Flow Bar Chart
    fig_bar = go.Figure()

    # Detect Premium / Discount
    if price > face_value:
        buy_color = "red"
        buy_annotation = "🔴 Premium Buy"
    elif price < face_value:
        buy_color = "green"
        buy_annotation = "🟢 Discount Buy"
    else:
        buy_color = "gray"
        buy_annotation = "⚪ Par Buy"

    # Update color for initial (BUY) cashflow only
    periodwise["Color"] = [
        buy_color if cashflow < 0 else ("seagreen" if idx == len(periodwise)-1 else "steelblue")
        for idx, cashflow in enumerate(periodwise["Cashflow"])
    ]

    # Prepare custom hover data
    periodwise["HoverLabel"] = [
        buy_annotation if cashflow < 0 else ("Coupon Payment" if idx != len(periodwise)-1 else "Final Coupon + Principal")
        for idx, cashflow in enumerate(periodwise["Cashflow"])
    ]

    # Plot bar with custom tooltips
    fig_bar.add_trace(go.Bar(
        x=periodwise["GroupLabel"],
        y=periodwise["Cashflow"],
        marker_color=periodwise["Color"],
        text=periodwise["Cashflow"].map(lambda x: f"₹{x:,.0f}"),
        textposition="outside",
        hovertemplate=
            "<b>Date:</b> %{x}<br>" +
            "<b>Cashflow:</b> ₹%{y:,.0f}<br>" +
            "<b>Type:</b> %{customdata[0]}<extra></extra>",
        customdata=periodwise[["HoverLabel"]]  # Custom Hover Labels
    ))

    fig_bar.update_layout(
        title="Cash Flow Breakdown (with Buy Type Indication)",
        height=320,
        xaxis_title="Period",
        yaxis_title="Cash Inflow/Outflow",
        hoverlabel=dict(bgcolor="white", font_size=12)
    )
    col1.plotly_chart(fig_bar, use_container_width=True)
    
    # Cumulative Return Chart
    fig_cum = go.Figure()
    fig_cum.add_trace(go.Scatter(
        x=cf_df["Date"],
        y=cf_df["Cumulative"],
        mode='lines+markers',
        name='Cumulative Return'
    ))

    # Determine Buy Type
    if price > face_value:
        annotation_text = "🔴 Premium Buy"
        annotation_color = "red"
    elif price < face_value:
        annotation_text = "🟢 Discount Buy"
        annotation_color = "green"
    else:
        annotation_text = "⚪ Par Buy"
        annotation_color = "gray"

    fig_cum.add_hline(
        y=price,
        line_dash="dot",
        line_color=annotation_color,
        annotation_text=annotation_text,
        annotation_position="top right",
        annotation_font_color=annotation_color
    )

    fig_cum.update_layout(
        title="Cumulative Return vs Investment",
        height=320,
        xaxis_title="Date",
        yaxis_title="₹"
    )
    col2.plotly_chart(fig_cum, use_container_width=True)

    # Table with heading
    with col3:
        st.markdown("#### 📋 Detailed Bond Cashflows")
        st.dataframe(cf_df.style.format({"Cashflow": "₹{:,.0f}", "Cumulative": "₹{:,.0f}"}), height=160)
