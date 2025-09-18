
# streamlit_l2o_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

st.set_page_config(page_title="L2O Profitability & Process Dashboard", layout="wide")

st.title("Lead-to-Order (L2O) — Profitability & Process Dashboard (Prototype)")
st.markdown("This prototype uses synthetic data. Replace `generate_data()` with your real data source.")

@st.cache_data
def generate_data(n=500, start_date="2025-01-01"):
    np.random.seed(42)
    start = datetime.fromisoformat(start_date)
    leads = []
    customers = [f"Cust_{i}" for i in range(1,11)]
    sectors = ["Retail","FMCG","Automotive","Electronics","Pharma"]
    regions = ["North","South","East","West"]
    routes = [f"City{a}-City{b}" for a in ["A","B","C","D"] for b in ["E","F","G","H"]]
    fleet_types = ["FTL","LTL","Reefer","Express"]
    for i in range(1, n+1):
        lead_date = start + timedelta(days=int(np.random.exponential(30)))
        customer = np.random.choice(customers, p=[0.15,0.15,0.12,0.12,0.1,0.08,0.08,0.08,0.07,0.05])
        sector = np.random.choice(sectors)
        region = np.random.choice(regions)
        route = np.random.choice(routes)
        distance_km = np.random.randint(50, 1500)
        weight_t = np.round(np.random.uniform(0.5, 25),1)
        service = np.random.choice(fleet_types, p=[0.4,0.3,0.2,0.1])
        # Lead -> Quote latency
        lead_to_quote_days = int(np.random.choice([0,1,2,3,4,5,7,10], p=[0.1,0.25,0.2,0.15,0.1,0.05,0.075,0.05]))
        quote_date = lead_date + timedelta(days=lead_to_quote_days)
        # Quote -> Order probability depends on margin and response time
        base_cost = distance_km * (0.6 if service=="FTL" else 0.75 if service=="LTL" else 1.2 if service=="Reefer" else 1.5)
        overhead = base_cost * 0.12
        estimated_cost = base_cost + overhead + np.random.normal(0, 20)
        # Market price signal
        market_multiplier = np.random.uniform(0.9, 1.2)
        quoted_price = max(estimated_cost * np.random.uniform(1.08, 1.30), estimated_cost + 50)
        # Discounts sometimes applied
        discount = 0.0
        if np.random.rand() < 0.18:
            discount = np.random.uniform(0.01, 0.25)
            quoted_price = quoted_price * (1 - discount)
        expected_margin = (quoted_price - estimated_cost) / quoted_price
        # Negotiation iterations
        negotiation_iters = np.random.poisson(0.6)
        # Approval rules simulated (some quotes get rejected)
        approval_flag = True
        approval_level = "Auto"
        if expected_margin < 0.10:
            if np.random.rand() < 0.6:
                approval_flag = False
                approval_level = "Rejected"
            else:
                approval_flag = True
                approval_level = "Manager"
        elif expected_margin < 0.13:
            approval_level = "Manager"
        # Quote outcome depends on price competitiveness and response time (faster wins more)
        win_prob = np.clip(0.65 + (expected_margin - 0.12) - (lead_to_quote_days * 0.03) + (market_multiplier-1)*0.5, 0.05, 0.95)
        won = np.random.rand() < win_prob
        quote_to_order_days = int(np.random.choice([0,1,2,3,5,7], p=[0.05,0.4,0.25,0.15,0.1,0.05]))
        order_date = quote_date + timedelta(days=quote_to_order_days) if won else None

        # If won, actuals may vary
        actual_cost = estimated_cost + np.random.normal(0, estimated_cost*0.05)
        # Simulate execution problems that add extra cost
        extra_cost = 0.0
        delay_flag = False
        extra_reason = None
        if won and np.random.rand() < 0.12:
            # penalty or extra handling
            extra_cost = estimated_cost * np.random.uniform(0.05, 0.25)
            actual_cost += extra_cost
            delay_flag = True
            extra_reason = np.random.choice(["Delay","Empty_Return","Damage","Customs"])
        actual_revenue = quoted_price if won else 0.0
        actual_margin = (actual_revenue - actual_cost)/actual_revenue if won and actual_revenue>0 else None

        leads.append({
            "Lead_ID": f"L{i:05d}",
            "Lead_Date": lead_date.date(),
            "Customer": customer,
            "Customer_Sector": sector,
            "Region": region,
            "Route": route,
            "Distance_km": distance_km,
            "Weight_t": weight_t,
            "Service_Type": service,
            "Lead_to_Quote_Days": lead_to_quote_days,
            "Quote_Date": quote_date.date(),
            "Estimated_Cost": round(float(estimated_cost),2),
            "Quoted_Price": round(float(quoted_price),2),
            "Discount": round(float(discount),3),
            "Expected_Margin": round(float(expected_margin),3),
            "Negotiation_Iterations": int(negotiation_iters),
            "Approval_Flag": approval_flag,
            "Approval_Level": approval_level,
            "Quote_Won": won,
            "Quote_to_Order_Days": quote_to_order_days if won else None,
            "Order_Date": order_date.date() if order_date else None,
            "Planned_Revenue": round(float(quoted_price),2) if won else 0.0,
            "Planned_Cost": round(float(estimated_cost),2) if won else 0.0,
            "Planned_Margin": round(((quoted_price - estimated_cost)/quoted_price) if won else 0,3),
            "Actual_Revenue": round(float(actual_revenue),2) if won else 0.0,
            "Actual_Cost": round(float(actual_cost),2) if won else 0.0,
            "Actual_Margin": round(float(actual_margin),3) if (won and actual_revenue>0) else None,
            "Delay_Flag": delay_flag,
            "Extra_Cost_Reason": extra_reason
        })
    df = pd.DataFrame(leads)
    # Time dims
    df["Lead_Month"] = pd.to_datetime(df["Lead_Date"]).dt.to_period("M").astype(str)
    df["Order_Month"] = pd.to_datetime(df["Order_Date"]).dt.to_period("M").astype(str)
    return df

df = generate_data(800)

# Sidebar filters
st.sidebar.header("Filters & Parameters")
date_min = st.sidebar.date_input("Leads since", value=pd.to_datetime(df["Lead_Date"]).min().date())
date_max = st.sidebar.date_input("Leads before", value=pd.to_datetime(df["Lead_Date"]).max().date())
selected_customers = st.sidebar.multiselect("Customer", options=sorted(df["Customer"].unique()), default=sorted(df["Customer"].unique()))
selected_regions = st.sidebar.multiselect("Region", options=sorted(df["Region"].unique()), default=sorted(df["Region"].unique()))
selected_service = st.sidebar.multiselect("Service Type", options=sorted(df["Service_Type"].unique()), default=sorted(df["Service_Type"].unique()))
margin_threshold = st.sidebar.slider("Alert Margin Threshold", min_value=0.0, max_value=0.3, value=0.12, step=0.01)

mask = (pd.to_datetime(df["Lead_Date"]) >= pd.to_datetime(date_min)) & (pd.to_datetime(df["Lead_Date"]) <= pd.to_datetime(date_max))
mask &= df["Customer"].isin(selected_customers)
mask &= df["Region"].isin(selected_regions)
mask &= df["Service_Type"].isin(selected_service)
fdf = df[mask].copy()

# Top KPIs
st.subheader("Executive KPIs")
col1, col2, col3, col4 = st.columns(4)
pipeline_value = fdf["Estimated_Cost"].sum() + (fdf["Estimated_Cost"].sum()*0.18)  # naive
confirmed_revenue = fdf["Planned_Revenue"].sum()
actual_revenue = fdf["Actual_Revenue"].sum()
avg_expected_margin = fdf["Expected_Margin"].dropna().mean()
avg_actual_margin = fdf["Actual_Margin"].dropna().mean()

col1.metric("Pipeline (est. potential)", f"${int(pipeline_value):,}")
col2.metric("Quoted / Planned Revenue", f"${int(confirmed_revenue):,}")
col3.metric("Actual Revenue (executed)", f"${int(actual_revenue):,}")
col4.metric("Avg Expected Margin", f"{avg_expected_margin:.1%}" if not np.isnan(avg_expected_margin) else "n/a")

# Funnel: Leads -> Quotes -> Orders (counts and revenue)
st.subheader("Process Funnel & Conversion Rates")
funnel_df = pd.DataFrame({
    "stage": ["Leads","Quotes Sent","Orders Confirmed"],
    "count": [len(fdf), fdf.shape[0], fdf[fdf["Quote_Won"]==True].shape[0]],
    "value": [fdf["Estimated_Cost"].sum(), fdf["Quoted_Price"].sum(), fdf["Planned_Revenue"].sum()]
})
fig_funnel = px.funnel(funnel_df, x='value', y='stage', title="Revenue Funnel (Est → Quoted → Confirmed)")
st.plotly_chart(fig_funnel, use_container_width=True)

# Expected vs Actual margin over time
st.subheader("Margin Trend & Distribution")
margins_time = fdf.groupby("Lead_Month").agg(
    expected_margin=("Expected_Margin","mean"),
    actual_margin=("Actual_Margin","mean"),
    planned_revenue=("Planned_Revenue","sum"),
    actual_revenue=("Actual_Revenue","sum")
).reset_index()
fig_margin = go.Figure()
fig_margin.add_trace(go.Scatter(x=margins_time["Lead_Month"], y=margins_time["expected_margin"], mode="lines+markers", name="Expected Margin"))
fig_margin.add_trace(go.Scatter(x=margins_time["Lead_Month"], y=margins_time["actual_margin"], mode="lines+markers", name="Actual Margin"))
fig_margin.update_layout(title="Expected vs Actual Margin (by Lead Month)", xaxis_title="Month", yaxis_title="Margin")
st.plotly_chart(fig_margin, use_container_width=True)

# Margin distribution histogram
fig_hist = px.histogram(fdf, x="Expected_Margin", nbins=30, title="Expected Margin Distribution (Quotes)")
st.plotly_chart(fig_hist, use_container_width=True)

# Bottleneck analysis: avg times
st.subheader("Process Efficiency & Bottlenecks")
avg_lead_to_quote = fdf["Lead_to_Quote_Days"].mean()
avg_quote_to_order = fdf["Quote_to_Order_Days"].dropna().mean()
st.write(f"Average Lead → Quote Days: **{avg_lead_to_quote:.1f}**")
st.write(f"Average Quote → Order Days: **{avg_quote_to_order:.1f}** (only won quotes)")

# Approval & delays
st.subheader("Approval Outcomes & Delays")
approval_counts = fdf["Approval_Level"].value_counts().reset_index().rename(columns={"index":"Approval_Level","Approval_Level":"count"})
st.bar_chart(approval_counts.set_index("Approval_Level")["count"])

delay_summary = fdf.groupby("Extra_Cost_Reason").agg(count=("Lead_ID","count")).reset_index()
st.bar_chart(delay_summary.set_index("Extra_Cost_Reason")["count"])

# Root cause analysis: slicers and heatmap
st.subheader("Root-Cause Drilldowns")
colA, colB = st.columns(2)
with colA:
    slicer_customer = st.selectbox("Drill by Customer", options=["All"] + sorted(fdf["Customer"].unique()), index=0)
with colB:
    slicer_route = st.selectbox("Drill by Route", options=["All"] + sorted(fdf["Route"].unique()), index=0)

drill = fdf.copy()
if slicer_customer != "All":
    drill = drill[drill["Customer"]==slicer_customer]
if slicer_route != "All":
    drill = drill[drill["Route"]==slicer_route]

# Profitability by route & service
profit_by_route = drill.groupby("Route").agg(avg_expected_margin=("Expected_Margin","mean"), avg_actual_margin=("Actual_Margin","mean"), count=("Lead_ID","count")).reset_index().sort_values("count", ascending=False).head(20)
st.write("Top Routes (by volume) — Avg Expected vs Actual Margin")
st.dataframe(profit_by_route.style.format({"avg_expected_margin":"{:.1%}","avg_actual_margin":"{:.1%}"}))

# Sales behavior view
st.subheader("Sales Behavior & Discounting")
sales_table = fdf.groupby("Customer").agg(count_quotes=("Lead_ID","count"), avg_discount=("Discount","mean"), avg_expected_margin=("Expected_Margin","mean")).reset_index().sort_values("count_quotes", ascending=False)
st.dataframe(sales_table.style.format({"avg_discount":"{:.2%}","avg_expected_margin":"{:.1%}"}))

# Alerts & Actionable Items
st.subheader("Alerts & Action Items (Profitability Gatekeeper)")
alerts = fdf[(fdf["Expected_Margin"] < margin_threshold) | (fdf["Discount"] > 0.15) | (fdf["Approval_Level"]=="Rejected")]
alerts_table = alerts[["Lead_ID","Lead_Date","Customer","Route","Service_Type","Quoted_Price","Estimated_Cost","Expected_Margin","Discount","Approval_Level","Quote_Won"]]
st.dataframe(alerts_table.sort_values("Expected_Margin").reset_index(drop=True))

st.markdown("### Suggested Actions (generated):")
suggestions = []
# Simple heuristic suggestions
if alerts.shape[0] > 0:
    # top reasons
    top_customers = alerts["Customer"].value_counts().head(3).to_dict()
    for c,r in top_customers.items():
        suggestions.append(f"- Review pricing and approval rules for **{c}**: {r} alerts flagged.")
    suggestions.append("- Update cost library for routes with frequent extra costs (check Extra_Cost_Reason).")
    suggestions.append("- Enforce approval gate for quotes below margin threshold or require manager sign-off.")
else:
    suggestions.append("- No critical alerts in the filtered set. Continue monitoring.")

for s in suggestions:
    st.write(s)

# Data export
st.subheader("Download Sample Dataset (CSV)")
@st.cache_data
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

csv = convert_df_to_csv(fdf)
st.download_button("Download filtered data as CSV", csv, "l2o_sample_data.csv", "text/csv")

st.markdown("---")
st.caption("Prototype for demonstration. Replace synthetic generator with your real data source and adjust thresholds & business rules.")
