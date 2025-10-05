# import standard libraries
import streamlit as st
import pandas as pd
import numpy as np

# Plotly
try:
    import plotly.express as px
    import plotly.graph_objects as go
    PLOTLY_OK = True
    # Set some plotly defaults
    px.defaults.template = "plotly_white"
    px.defaults.width = None
    px.defaults.height = 420
except Exception:
    PLOTLY_OK = False

st.set_page_config(page_title="ORD Flight Difficulty Score", layout="wide")
st.set_option("client.showErrorDetails", True)

# ---------- Utilities ----------
KEY_COLS = [
    "company_id",
    "flight_number",
    "scheduled_departure_date_local",
    "scheduled_departure_station_code",
    "scheduled_arrival_station_code",
]

def ensure_columns(df: pd.DataFrame, cols, fill="") -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c not in df.columns:
            df[c] = fill
    return df

def to_date_series(s):
    return pd.to_datetime(s, errors="coerce")

def first_present(candidates, columns):
    for c in candidates:
        if c in columns:
            return c
    return None

def sanitize_color(df: pd.DataFrame, choice: str | None) -> str | None:
    """Ensure the selected color column exists and isn't 'class' for EDA."""
    if choice is None:
        return None
    if choice == "None":
        return None
    if choice == "class":
        return None
    return choice if choice in df.columns else None

def tidy_layout(fig, x_title=None, y_title=None, title=None):
    fig.update_layout(
        title=dict(text=title, x=0.01, xanchor="left") if title else None,
        margin=dict(l=10, r=10, t=50, b=10),
        legend=dict(orientation="h", y=1.08, x=0.0),
        hoverlabel=dict(namelength=-1),
    )
    if x_title:
        fig.update_xaxes(title=x_title, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    else:
        fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    if y_title:
        fig.update_yaxes(title=y_title, showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    else:
        fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.08)")
    return fig

# Reading data
@st.cache_data
def load_data():
    flights = pd.read_csv(
        "Flight Level Data.csv",
        parse_dates=[
            "scheduled_departure_datetime_local",
            "scheduled_arrival_datetime_local",
            "actual_departure_datetime_local",
            "actual_arrival_datetime_local",
        ],
        dayfirst=False,
        infer_datetime_format=True,
    )
    bags = pd.read_csv("Bag+Level+Data.csv")
    pnr_flight = pd.read_csv("PNR+Flight+Level+Data.csv")
    try:
        pnr_remarks = pd.read_csv("PNR Remark Level Data.csv")
    except Exception:
        pnr_remarks = pd.DataFrame(columns=KEY_COLS + ["remark_text"])
    airports = pd.read_csv("Airports Data.csv")
    return flights, bags, pnr_flight, pnr_remarks, airports

# feature engineering
def engineer_features(flights, bags, pnr_flight, pnr_remarks, airports, margin_threshold=5):
    f = flights.copy()

    for col in [
        "scheduled_departure_datetime_local",
        "scheduled_arrival_datetime_local",
        "actual_departure_datetime_local",
        "actual_arrival_datetime_local",
    ]:
        if col not in f.columns:
            f[col] = pd.NaT

    if "scheduled_departure_date_local" not in f.columns or f["scheduled_departure_date_local"].isna().all():
        f["scheduled_departure_date_local"] = (
            f["scheduled_departure_datetime_local"].dt.date.astype("string")
        )

    f["dep_delay_min"] = (
        (f["actual_departure_datetime_local"] - f["scheduled_departure_datetime_local"])
        .dt.total_seconds()
        .div(60.0)
    )
    f["arr_delay_min"] = (
        (f["actual_arrival_datetime_local"] - f["scheduled_arrival_datetime_local"])
        .dt.total_seconds()
        .div(60.0)
    )

    if "scheduled_ground_time_minutes" not in f.columns:
        f["scheduled_ground_time_minutes"] = 0
    if "minimum_turn_minutes" not in f.columns:
        f["minimum_turn_minutes"] = 0

    f["ground_time_margin"] = f["scheduled_ground_time_minutes"] - f["minimum_turn_minutes"]
    f["dep_date"] = to_date_series(f["scheduled_departure_date_local"])

    f = ensure_columns(f, KEY_COLS, fill="")
    bags = ensure_columns(bags, KEY_COLS, fill="")
    pnr_flight = ensure_columns(pnr_flight, KEY_COLS, fill="")
    pnr_remarks = ensure_columns(pnr_remarks, KEY_COLS + ["remark_text"], fill="")

    # Bags
    b = bags.copy()
    if "bag_type" not in b.columns:
        b["bag_type"] = ""
    if "bag_tag_unique_number" not in b.columns:
        b["bag_tag_unique_number"] = ""
    b["is_transfer"] = b["bag_type"].astype(str).str.contains("Transfer", case=False, na=False)
    bag_agg = (
        b.groupby(KEY_COLS, dropna=False)
        .agg(total_bags=("bag_tag_unique_number", "count"), transfer_bags=("is_transfer", "sum"))
        .reset_index()
    )
    bag_agg["transfer_ratio"] = np.where(
        bag_agg["total_bags"] > 0, bag_agg["transfer_bags"] / bag_agg["total_bags"], 0.0
    )

    # Pax
    pf = pnr_flight.copy()
    if "total_pax" not in pf.columns:
        pf["total_pax"] = 0
    if "is_child" not in pf.columns:
        pf["is_child"] = "N"
    pax_agg = (
        pf.groupby(KEY_COLS, dropna=False)
        .agg(pax=("total_pax", "sum"), child_pax=("is_child", lambda s: (s == "Y").sum()))
        .reset_index()
    )

    # SSR
    ssr_codes = ["WCHR", "WCHS", "WCHC", "UMNR", "MEDA", "BLND", "DEAF"]
    r = pnr_remarks.copy()
    if "remark_text" not in r.columns:
        r["remark_text"] = ""
    r["remark_text"] = r["remark_text"].astype(str)
    if not r.empty:
        for code in ssr_codes:
            r[code] = r["remark_text"].str.contains(code, case=False, na=False).astype(int)
        ssr_agg = (
            r.groupby(KEY_COLS, dropna=False)
             .agg(**{f"SSR_{c}": (c, "sum") for c in ssr_codes})
             .reset_index()
        )
        ssr_agg["SSR_total"] = ssr_agg[[f"SSR_{c}" for c in ssr_codes]].sum(axis=1)
    else:
        ssr_agg = pd.DataFrame(columns=KEY_COLS + ["SSR_total"] + [f"SSR_{c}" for c in ssr_codes])

    merged = (
        f.merge(pax_agg, on=KEY_COLS, how="left")
         .merge(bag_agg, on=KEY_COLS, how="left")
         .merge(ssr_agg, on=KEY_COLS, how="left")
    )

    # Airports
    ac = airports.copy()
    if "airport_iata_code" in ac.columns:
        ac = ac.rename(columns={"airport_iata_code": "scheduled_arrival_station_code"})
        merged = merged.merge(ac, on="scheduled_arrival_station_code", how="left")
    else:
        merged["iso_country_code"] = np.nan

    if "iso_country_code" not in merged.columns:
        merged["iso_country_code"] = np.nan
    merged["is_international"] = (merged["iso_country_code"].fillna("") != "US").astype(int)

    if "total_seats" not in merged.columns:
        merged["total_seats"] = np.nan
    merged["pax"] = merged["pax"].fillna(0)
    merged["total_seats"] = merged["total_seats"].replace({0: np.nan})
    merged["load_factor"] = merged["pax"] / merged["total_seats"]
    merged["transfer_ratio"] = merged["transfer_ratio"].fillna(0)
    merged["total_bags"] = merged["total_bags"].fillna(0)
    merged["SSR_total"] = merged["SSR_total"].fillna(0)

    merged["ground_risk"] = -(merged["scheduled_ground_time_minutes"] - merged["minimum_turn_minutes"])
    merged["close_or_below_turn"] = (merged["ground_time_margin"] <= margin_threshold).astype(int)

    return merged

# scoring
def daily_scores(df: pd.DataFrame, weights: dict) -> pd.DataFrame:
    def zscore(s: pd.Series) -> pd.Series:
        std = s.std(ddof=0)
        if std == 0 or np.isnan(std):
            return pd.Series(0.0, index=s.index)
        return (s - s.mean()) / std

    features = list(weights.keys())
    base = df.dropna(subset=["dep_date"]).copy()
    parts = []

    for _, g in base.groupby("dep_date"):
        z = pd.DataFrame(index=g.index)
        for ftr in features:
            z[ftr] = zscore(g[ftr].fillna(0))

        score = sum(weights[f] * z[f] for f in features)
        rank = (-score).rank(method="dense").astype(int)

        n = len(rank)
        cut1 = int(np.ceil(n / 3))
        cut2 = int(np.ceil(2 * n / 3))
        cls = np.where(rank <= cut1, "Difficult", np.where(rank <= cut2, "Medium", "Easy"))

        tmp = g.copy()
        tmp["difficulty_score"] = score
        tmp["rank_in_day"] = rank
        tmp["class"] = cls
        parts.append(tmp)

    if parts:
        out = pd.concat(parts).sort_values(["dep_date", "rank_in_day"])
    else:
        out = base.copy()
        out["difficulty_score"] = np.nan
        out["rank_in_day"] = np.nan
        out["class"] = np.nan
    return out

st.title("ORD Flight Difficulty Score")
st.caption("Two-week ORD departures • Daily-reset scoring • Built by TEAMDA")
with st.sidebar:
    st.header("Navigator")
    section = st.radio(
        "Go to",
        ["1) Exploratory Data Analysis (EDA)", "2) Flight Difficulty Score", "3) Post-Analysis & Operational Insights"],
        index=0,
    )

    st.divider()
    st.header("Controls")
    margin_threshold = st.slider("Close-to-min turn threshold (min)", 0, 60, 5, 1)

    st.markdown("**Weights** (sum ~ 1)")
    w_ground = st.number_input("Ground Risk", value=0.35, step=0.01)
    w_transfer = st.number_input("Transfer Ratio", value=0.30, step=0.01)
    w_load = st.number_input("Load Factor", value=0.25, step=0.01)
    w_ssr = st.number_input("SSR Total", value=0.10, step=0.01)
    weights = {
        "ground_risk": float(w_ground),
        "transfer_ratio": float(w_transfer),
        "load_factor": float(w_load),
        "SSR_total": float(w_ssr),
    }

# Loading data and building
flights, bags, pnr_flight, pnr_remarks, airports = load_data()
df = engineer_features(flights, bags, pnr_flight, pnr_remarks, airports, margin_threshold)
scores = daily_scores(df, weights)

with st.sidebar:
    st.divider()
    st.header("Global Filters")

    all_dates = pd.to_datetime(df["dep_date"].dropna().unique())
    if len(all_dates):
        d_min, d_max = all_dates.min(), all_dates.max()
        date_range = st.date_input("Departure date range", (d_min, d_max))
        if isinstance(date_range, tuple) and len(date_range) == 2:
            d_from, d_to = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
            df = df[(df["dep_date"] >= d_from) & (df["dep_date"] <= d_to)]
            scores = scores[(scores["dep_date"] >= d_from) & (scores["dep_date"] <= d_to)]
    else:
        st.caption("No dep_date values to filter.")

    carriers = sorted([c for c in df["company_id"].dropna().unique() if str(c) != ""])
    carrier_sel = st.multiselect("Company/Airline", carriers, default=carriers[:5] if carriers else [])
    if carrier_sel:
        df = df[df["company_id"].isin(carrier_sel)]
        scores = scores[scores["company_id"].isin(carrier_sel)]

    dep_stations = sorted([c for c in df["scheduled_departure_station_code"].dropna().unique()])
    arr_stations = sorted([c for c in df["scheduled_arrival_station_code"].dropna().unique()])
    dep_sel = st.multiselect("Departure stations", dep_stations)
    arr_sel = st.multiselect("Arrival stations", arr_stations)

    if dep_sel:
        df = df[df["scheduled_departure_station_code"].isin(dep_sel)]
        scores = scores[scores["scheduled_departure_station_code"].isin(dep_sel)]
    if arr_sel:
        df = df[df["scheduled_arrival_station_code"].isin(arr_sel)]
        scores = scores[scores["scheduled_arrival_station_code"].isin(arr_sel)]

    clip_low, clip_high = st.slider("Clip departure delay (min)", -120, 240, (-60, 180), 5)

def require_plotly():
    if not PLOTLY_OK:
        st.warning("Plotly is not available in this environment. Install `plotly` for interactive charts.")
        return False
    return True

# DELIVERABLE 1: EDA 
if section.startswith("1"):
    st.subheader("Key Flight-Level KPIs")
    c1, c2, c3 = st.columns(3)
    c1.metric("Avg Departure Delay (min)", f"{df['dep_delay_min'].mean():.1f}")
    c2.metric("% Departed Late", f"{(df['dep_delay_min'] > 0).mean() * 100:.1f}%")
    c3.metric("# Close/Below Min Turn", int(df["close_or_below_turn"].sum()))

    c4, c5, c6 = st.columns(3)
    c4.metric("Avg Transfer Bag Ratio", f"{df['transfer_ratio'].mean() * 100:.1f}%")
    mean_lf = df["load_factor"].mean(skipna=True)
    c5.metric("Avg Load Factor", f"{(mean_lf * 100 if pd.notna(mean_lf) else 0):.1f}%")
    c6.metric("Avg SSR per Flight", f"{df['SSR_total'].mean():.2f}")

    if require_plotly():
        st.markdown("### Distributions & Relationships")

        # Histogram: Departure delays 
        dfx = df.copy()
        dfx["dep_delay_min_clip"] = dfx["dep_delay_min"].clip(lower=clip_low, upper=clip_high)
        bins = st.slider("Histogram bins", 10, 100, 40, 5)

        eda_color_candidates = ["scheduled_arrival_station_code", "company_id", "is_international"]
        eda_color_choices = ["None"] + [c for c in eda_color_candidates if c in dfx.columns]
        eda_color_default = "None"
        color_by_hist = st.selectbox("Color histogram by", eda_color_choices, index=eda_color_choices.index(eda_color_default))
        color_kw_hist = sanitize_color(dfx, color_by_hist)

        fig_hist = px.histogram(
            dfx,
            x="dep_delay_min_clip",
            nbins=bins,
            color=color_kw_hist,
            opacity=0.9,
            hover_data=["company_id","flight_number","scheduled_departure_station_code","scheduled_arrival_station_code"],
        )
        fig_hist.update_traces(marker_line_width=0.2, marker_line_color="rgba(0,0,0,0.3)")
        fig_hist = tidy_layout(
            fig_hist,
            x_title=f"Departure Delay (min)  | clipped [{clip_low}, {clip_high}]",
            y_title="Flights",
            title="Departure Delay Distribution",
        )
        st.plotly_chart(fig_hist, use_container_width=True)

        # Scatter: Load vs Delay 
        st.markdown("### Load Factor vs Departure Delay")
        ptsize = st.slider("Point size", 4, 18, 8)
        scatter_color_candidates = ["scheduled_arrival_station_code", "company_id", "is_international"]
        scatter_color_choices = [c for c in scatter_color_candidates if c in dfx.columns]
        
        default_idx = 0 if scatter_color_choices else None
        color_by_scatter = st.selectbox(
            "Color scatter by",
            scatter_color_choices if scatter_color_choices else ["(none)"],
            index=default_idx if default_idx is not None else 0,
        )
        color_kw_scatter = sanitize_color(dfx, color_by_scatter if scatter_color_choices else None)

        dfy = dfx.dropna(subset=["load_factor","dep_delay_min"])
        fig_sc = px.scatter(
            dfy,
            x="load_factor",
            y="dep_delay_min",
            color=color_kw_scatter,
            hover_data=["company_id","flight_number","SSR_total","transfer_ratio","ground_time_margin"],
        )
        fig_sc.update_traces(marker=dict(size=ptsize, opacity=0.85, line=dict(width=0.5, color="rgba(0,0,0,0.25)")))
        fig_sc = tidy_layout(
            fig_sc,
            x_title="Load Factor",
            y_title="Departure Delay (min)",
            title="Load vs Delay (interactive)",
        )
        st.plotly_chart(fig_sc, use_container_width=True)

        st.markdown("### Delay spread by destination (top N by flights)")
        topN = st.slider("Show top N destinations", 5, 30, 12, 1)
        top_dest = (df["scheduled_arrival_station_code"]
                    .value_counts()
                    .head(topN)
                    .index.tolist())
        vdf = dfx[dfx["scheduled_arrival_station_code"].isin(top_dest)].copy()
        vdf["dep_delay_min_clip"] = vdf["dep_delay_min"].clip(lower=clip_low, upper=clip_high)
        if len(vdf):
            fig_v = px.violin(
                vdf,
                x="scheduled_arrival_station_code",
                y="dep_delay_min_clip",
                box=True,
                points="suspectedoutliers",
                hover_data=["company_id","flight_number"],
            )
            fig_v = tidy_layout(
                fig_v,
                x_title="Destination",
                y_title=f"Departure Delay (min) | clipped [{clip_low}, {clip_high}]",
                title="Delay Distribution by Destination",
            )
            st.plotly_chart(fig_v, use_container_width=True)

        #  Heatmap correlations
        st.markdown("### Feature correlations")
        feat_cols = ["dep_delay_min","arr_delay_min","ground_time_margin","ground_risk","transfer_ratio","load_factor","SSR_total"]
        use_cols = [c for c in feat_cols if c in df.columns]
        if len(use_cols) >= 2:
            corr = df[use_cols].corr(numeric_only=True)
            fig_hm = px.imshow(
                corr,
                text_auto=".2f",
                aspect="auto",
            )
            fig_hm = tidy_layout(fig_hm, title="Correlation Heatmap")
            st.plotly_chart(fig_hm, use_container_width=True)

        # Bar: SSR composition 
        st.markdown("### SSR composition (total over filter)")
        ssr_cols = [c for c in df.columns if c.startswith("SSR_") and c != "SSR_total"]
        if ssr_cols:
            ssr_sum = df[ssr_cols].sum().reset_index()
            ssr_sum.columns = ["SSR_Code","count"]
            fig_ssr = px.bar(
                ssr_sum,
                x="SSR_Code",
                y="count",
                text="count",
            )
            fig_ssr.update_traces(textposition="outside", cliponaxis=False)
            fig_ssr = tidy_layout(fig_ssr, x_title="SSR Code", y_title="Count", title="Total SSR Counts")
            st.plotly_chart(fig_ssr, use_container_width=True)

    st.markdown("#### Preview: Flight-Level Data")
    st.dataframe(df.head(100), use_container_width=True, height=360)

    st.download_button(
        "Download Cleaned Flight Features (CSV)",
        data=df.to_csv(index=False).encode("utf-8"),
        file_name="ord_flight_features.csv",
        mime="text/csv",
    )

# DELIVERABLE 2: Difficulty Score 
elif section.startswith("2"):
    st.subheader("Daily Difficulty Rankings")

    all_days = sorted(pd.to_datetime(scores["dep_date"].dropna().unique()))
    if all_days:
        day = st.selectbox("Pick a date", all_days, index=len(all_days) - 1)
        day_tbl = (
            scores[scores["dep_date"] == day][
                [
                    "rank_in_day",
                    "class",
                    "company_id",
                    "flight_number",
                    "scheduled_departure_station_code",
                    "scheduled_arrival_station_code",
                    "difficulty_score",
                    "dep_delay_min",
                    "pax",
                    "total_seats",
                    "load_factor",
                    "transfer_ratio",
                    "SSR_total",
                    "ground_time_margin",
                    "minimum_turn_minutes",
                    "scheduled_ground_time_minutes",
                ]
            ]
            .sort_values("rank_in_day")
            .reset_index(drop=True)
        )
        st.dataframe(day_tbl, use_container_width=True, height=420)

        if require_plotly() and len(day_tbl):
            st.markdown("##### Interactive Rank Bar")
            fig_rank = px.bar(
                day_tbl,
                x="rank_in_day",
                y="difficulty_score",
                color="class",  
                hover_data=["company_id","flight_number","scheduled_arrival_station_code","load_factor","transfer_ratio","SSR_total"],
            )
            fig_rank = tidy_layout(fig_rank, x_title="Rank (1 = hardest)", y_title="Difficulty Score", title=f"Difficulty Score by Rank — {pd.to_datetime(day).date()}")
            st.plotly_chart(fig_rank, use_container_width=True)

    else:
        st.info("No dates available for scoring. Check input data.")

    st.markdown("##### Destination Difficulty (aggregate)")
    dest = (
        scores.groupby("scheduled_arrival_station_code", dropna=False)
        .agg(
            avg_rank=("rank_in_day", "mean"),
            pct_difficult=("class", lambda s: (s == "Difficult").mean() * 100),
            n_flights=("rank_in_day", "count"),
        )
        .reset_index()
        .sort_values("pct_difficult", ascending=False)
    )
    st.dataframe(dest.head(50), use_container_width=True, height=360)

    if require_plotly() and len(dest):
        fig_dest = px.bar(
            dest.head(25),
            x="scheduled_arrival_station_code",
            y="pct_difficult",
            color="avg_rank",
            hover_data=["n_flights"],
        )
        fig_dest = tidy_layout(fig_dest, x_title="Destination", y_title="% Flights marked Difficult", title="Top Destinations by % Difficult")
        st.plotly_chart(fig_dest, use_container_width=True)

    st.markdown("##### What's driving 'Difficult' vs 'Easy'? (feature averages)")
    feat_cols = ["ground_risk", "transfer_ratio", "load_factor", "SSR_total", "dep_delay_min"]
    class_prof = (
        scores.groupby("class", dropna=False)[feat_cols]
        .mean(numeric_only=True)
        .reindex(["Difficult", "Medium", "Easy"])
    )
    st.dataframe(class_prof, use_container_width=True)

    if require_plotly() and class_prof.dropna(how="all").shape[0] > 0:
        cp = class_prof.reset_index().melt(id_vars="class", var_name="feature", value_name="value")
        fig_cp = px.bar(cp, x="feature", y="value", color="class", barmode="group")
        fig_cp = tidy_layout(fig_cp, x_title="Feature", y_title="Average value", title="Feature Profiles by Class")
        st.plotly_chart(fig_cp, use_container_width=True)

    st.download_button(
        "Download Daily Scores (CSV)",
        data=scores.to_csv(index=False).encode("utf-8"),
        file_name="ord_daily_scores.csv",
        mime="text/csv",
    )

# DELIVERABLE 3: Post-Analysis & Insights
else:
    st.subheader("Destinations That Consistently Skew Difficult")
    dest = (
        scores.groupby("scheduled_arrival_station_code", dropna=False)
        .agg(
            avg_rank=("rank_in_day", "mean"),
            pct_difficult=("class", lambda s: (s == "Difficult").mean() * 100),
            n_flights=("rank_in_day", "count"),
            avg_ground_risk=("ground_risk", "mean"),
            avg_transfer_ratio=("transfer_ratio", "mean"),
            avg_load=("load_factor", "mean"),
            avg_ssr=("SSR_total", "mean"),
            avg_dep_delay=("dep_delay_min", "mean"),
        )
        .reset_index()
    )

    focus = (
        dest[dest["n_flights"] >= 10]
        .sort_values(["pct_difficult", "avg_rank"], ascending=[False, True])
        .head(15)
        .reset_index(drop=True)
    )
    st.dataframe(focus, use_container_width=True, height=360)

    if require_plotly() and len(focus):
        fig_focus = px.scatter(
            focus,
            x="avg_rank",
            y="pct_difficult",
            size="n_flights",
            color="avg_ground_risk",
            hover_data=["scheduled_arrival_station_code","avg_transfer_ratio","avg_load","avg_ssr","avg_dep_delay"],
        )
        fig_focus = tidy_layout(fig_focus, x_title="Average Rank (lower = harder)", y_title="% Difficult", title="Focus Destinations (size=n flights, color=avg ground risk)")
        st.plotly_chart(fig_focus, use_container_width=True)

    # Optional map
    lat_cols = [c for c in airports.columns if "lat" in c.lower()]
    lon_cols = [c for c in airports.columns if "lon" in c.lower() or "lng" in c.lower()]
    if require_plotly() and ("scheduled_arrival_station_code" in airports.columns or "airport_iata_code" in airports.columns):
        ac = airports.copy()
        if "airport_iata_code" in ac.columns:
            ac = ac.rename(columns={"airport_iata_code": "scheduled_arrival_station_code"})
        lat = first_present(lat_cols, ac.columns)
        lon = first_present(lon_cols, ac.columns)
        if lat and lon:
            geo = ac[["scheduled_arrival_station_code", lat, lon]].dropna()
            geo = geo.merge(dest, on="scheduled_arrival_station_code", how="inner")
            if len(geo):
                fig_map = px.scatter_mapbox(
                    geo,
                    lat=lat,
                    lon=lon,
                    color="pct_difficult",
                    size="n_flights",
                    hover_name="scheduled_arrival_station_code",
                    hover_data={"avg_rank": True, "pct_difficult": True, "n_flights": True},
                    color_continuous_scale="RdYlGn_r",
                    zoom=2,
                    height=520,
                )
                fig_map.update_layout(mapbox_style="open-street-map", margin=dict(l=0, r=0, t=60, b=0), title=dict(text="Destination Difficulty Map", x=0.01))
                st.plotly_chart(fig_map, use_container_width=True)

    st.markdown("#### Common Drivers (heuristic signals)")
    med_gr = dest["avg_ground_risk"].median()
    med_tr = dest["avg_transfer_ratio"].median()
    med_ld = dest["avg_load"].median()
    med_ss = dest["avg_ssr"].median()

    driver_rows = []
    for _, r in focus.iterrows():
        drivers = []
        if pd.notna(r["avg_ground_risk"]) and r["avg_ground_risk"] > med_gr:
            drivers.append("tight or sub-optimal turns")
        if pd.notna(r["avg_transfer_ratio"]) and r["avg_transfer_ratio"] > med_tr:
            drivers.append("high transfer bag complexity")
        if pd.notna(r["avg_load"]) and r["avg_load"] > med_ld:
            drivers.append("consistently high load factors")
        if pd.notna(r["avg_ssr"]) and r["avg_ssr"] > med_ss:
            drivers.append("higher special service requests")
        if not drivers:
            drivers.append("mixed factors; explore crew/asset positioning")
        driver_rows.append((r["scheduled_arrival_station_code"], drivers))

    for code, drivers in driver_rows:
        st.markdown(f"- **{code}** → probable drivers: " + ", ".join(drivers))

    st.markdown("#### Recommended Actions")
    st.markdown(
        """
- **Safeguard minimum turns** on lowest-ranked days: add gate staff/ramp floaters, pre-stage kits, defer non-critical turn work.
- **Transfer spike playbook**: deploy transfer-bag sweeper during banks with high transfer ratios; monitor MCT exposures.
- **Load-driven delays**: start boarding earlier with dual-lane boarding; pre-assign SSR seats; streamline seat-change handling.
- **SSR planning**: pre-alert wheelchairs/medical assistance; stage aisle chairs; coordinate cabin crew for pre-boards.
- **SOPs by destination**: for top 5 challenging destinations, publish a 1-pager of local oddities, gate availability, peak transfer feeds.
"""
    )

    st.download_button(
        "Download Destination Difficulty Summary (CSV)",
        data=focus.to_csv(index=False).encode("utf-8"),
        file_name="ord_destination_difficulty_summary.csv",
        mime="text/csv",
    )
    
    
st.markdown(
    """
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: transparent;
        color: grey;
        text-align: center;
        font-size: 14px;
        padding: 10px 0;
        z-index: 1000;
    }
    </style>
    <div class="footer">
        <hr style="margin-bottom: 8px; border-color: #ddd;" />
        All Rights Reserved. Created by Akash Kumar Rajak & Mayank Yadav (TEAMDA)
    </div>
    """,
    unsafe_allow_html=True,
)