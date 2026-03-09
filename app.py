import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="COVID-19 Dashboard",
    page_icon=":microbe:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DATA_FILENAME = "owid-covid-data.csv"


def resolve_data_path() -> tuple[Path | None, list[Path]]:
    # Support both Linux-style (/mnt/data) and local project paths.
    candidates = [
        Path("/data") / DATA_FILENAME,
        Path("data") / DATA_FILENAME,
        Path(__file__).resolve().parent / "data" / DATA_FILENAME,
        Path.cwd() / "data" / DATA_FILENAME,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate, candidates
    return None, candidates

METRIC_LABELS = {
    "total_cases": "Total Cases",
    "new_cases": "New Cases",
    "new_cases_smoothed": "New Cases (Smoothed)",
    "total_deaths": "Total Deaths",
    "new_deaths": "New Deaths",
    "new_deaths_smoothed": "New Deaths (Smoothed)",
    "total_tests": "Total Tests",
    "new_tests": "New Tests",
    "people_vaccinated": "People Vaccinated",
    "people_fully_vaccinated": "People Fully Vaccinated",
    "total_vaccinations": "Total Vaccinations",
    "total_boosters": "Total Boosters",
    "icu_patients": "ICU Patients",
    "hosp_patients": "Hospital Patients",
    "reproduction_rate": "Reproduction Rate",
    "positive_rate": "Positive Rate",
    "stringency_index": "Stringency Index",
}

PER_CAPITA_METRICS = {
    "total_cases_per_million",
    "new_cases_per_million",
    "new_cases_smoothed_per_million",
    "total_deaths_per_million",
    "new_deaths_per_million",
    "new_deaths_smoothed_per_million",
    "icu_patients_per_million",
    "hosp_patients_per_million",
    "weekly_icu_admissions_per_million",
    "weekly_hosp_admissions_per_million",
    "new_vaccinations_smoothed_per_million",
}

FLOW_SUM_METRICS = {"new_cases", "new_deaths", "new_tests", "new_vaccinations"}


@st.cache_data(show_spinner=False)
def load_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["date"])
    df = df.sort_values(["location", "date"]).reset_index(drop=True)
    return df


@st.cache_data(show_spinner=False)
def get_latest_snapshot(df: pd.DataFrame) -> pd.DataFrame:
    base = df[df["continent"].notna()].copy()
    latest = base.groupby("location", as_index=False).tail(1)
    return latest


@st.cache_data(show_spinner=False)
def get_geo_frame(df: pd.DataFrame, as_of_date: pd.Timestamp, metric: str) -> pd.DataFrame:
    base = df[(df["continent"].notna()) & (df["date"] <= as_of_date)].copy()
    # Keep the latest non-null metric row per country (not just the latest row overall).
    base = base[base[metric].notna()]
    latest = base.groupby("location", as_index=False).tail(1)
    latest = latest.dropna(subset=["iso_code"])
    latest[metric] = latest[metric].clip(lower=0)
    return latest


@st.cache_data(show_spinner=False)
def available_locations(df: pd.DataFrame, continents: tuple[str, ...]) -> list[str]:
    base = df[df["continent"].notna()]
    if continents:
        base = base[base["continent"].isin(list(continents))]
    return sorted(base["location"].dropna().unique().tolist())


@st.cache_data(show_spinner=False)
def get_latest_in_range(df: pd.DataFrame, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
    base = df[(df["continent"].notna()) & (df["date"].between(start_date, end_date))].copy()
    latest = base.groupby("location", as_index=False).tail(1)
    return latest


@st.cache_data(show_spinner=False)
def get_latest_in_range_for_metric(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    metric: str,
) -> pd.DataFrame:
    base = df[(df["continent"].notna()) & (df["date"].between(start_date, end_date))].copy()
    base = base[base[metric].notna()]
    latest = base.groupby("location", as_index=False).tail(1)
    return latest


@st.cache_data(show_spinner=False)
def get_metric_snapshot(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    metric: str,
) -> pd.DataFrame:
    base = df[(df["continent"].notna()) & (df["date"].between(start_date, end_date))].copy()
    base = base.sort_values(["location", "date"])

    if metric in FLOW_SUM_METRICS:
        latest_meta = base.groupby("location", as_index=False).tail(1)[
            ["location", "continent", "iso_code", "total_cases", "total_deaths", "people_fully_vaccinated"]
        ]
        period_sum = base.groupby("location", as_index=False)[metric].sum(min_count=1)
        snap = latest_meta.merge(period_sum, on="location", how="left")
    else:
        metric_base = base[base[metric].notna()].copy()
        snap = metric_base.groupby("location", as_index=False).tail(1)

    snap = snap.dropna(subset=["iso_code"])
    if metric in snap.columns:
        snap[metric] = snap[metric].clip(lower=0)
    return snap


@st.cache_data(show_spinner=False)
def get_first_last_values(
    df: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    metric: str,
    locations: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    base = df[(df["continent"].notna()) & (df["date"].between(start_date, end_date))].copy()
    if locations:
        base = base[base["location"].isin(list(locations))]
    first_vals = base.groupby("location", as_index=False).head(1)[["location", metric]].rename(columns={metric: "start_value"})
    last_vals = base.groupby("location", as_index=False).tail(1)[["location", metric]].rename(columns={metric: "end_value"})
    return first_vals, last_vals


def fmt_num(value, decimals: int = 0) -> str:
    if value is None or (isinstance(value, float) and (math.isnan(value) or math.isinf(value))):
        return "-"
    if pd.isna(value):
        return "-"
    value = float(value)
    abs_val = abs(value)
    if abs_val >= 1_000_000_000:
        return f"{value/1_000_000_000:.2f}B"
    if abs_val >= 1_000_000:
        return f"{value/1_000_000:.2f}M"
    if abs_val >= 1_000:
        return f"{value/1_000:.2f}K"
    return f"{value:,.{decimals}f}"


CSS = """
<style>
    .stApp {
        background: linear-gradient(180deg, #08111f 0%, #050a14 100%);
        color: #f3f6fb;
    }
    .block-container {
        padding-top: 1.2rem;
        padding-bottom: 1rem;
        max-width: 1600px;
    }
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0b1526 0%, #08111f 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    .hero {
        padding: 1rem 1.2rem;
        border-radius: 20px;
        background: linear-gradient(135deg, rgba(19,31,53,0.95), rgba(9,16,28,0.95));
        border: 1px solid rgba(120,160,255,0.15);
        box-shadow: 0 10px 30px rgba(0,0,0,0.25);
        margin-bottom: 1rem;
    }
    .hero h1 {
        margin: 0;
        font-size: 2rem;
        color: #ffffff;
    }
    .hero p {
        margin: 0.35rem 0 0 0;
        color: #afbdd5;
        font-size: 1rem;
    }
    .kpi-card {
        background: linear-gradient(180deg, rgba(19,24,34,0.95), rgba(11,16,25,0.95));
        padding: 1rem 1.1rem;
        border-radius: 18px;
        border: 1px solid rgba(255,255,255,0.06);
        box-shadow: 0 12px 24px rgba(0,0,0,0.22);
        min-height: 120px;
    }
    .kpi-label {
        color: #aeb7c8;
        font-size: 0.95rem;
        margin-bottom: 0.4rem;
    }
    .kpi-value {
        color: #ff5a4e;
        font-size: 2rem;
        font-weight: 700;
        line-height: 1.1;
    }
    .kpi-sub {
        color: #90ee90;
        font-size: 0.9rem;
        margin-top: 0.5rem;
    }
    div[data-testid="stMetric"] {
        background: linear-gradient(180deg, rgba(19,24,34,0.95), rgba(11,16,25,0.95));
        border: 1px solid rgba(255,255,255,0.06);
        padding: 0.6rem 0.9rem;
        border-radius: 18px;
    }
    .panel-title {
        font-size: 1.05rem;
        font-weight: 700;
        color: #f5f8fd;
        margin-bottom: 0.55rem;
    }
</style>
"""

st.markdown(CSS, unsafe_allow_html=True)

data_path, checked_paths = resolve_data_path()
if data_path is None:
    checked = "\n".join(f"- `{p}`" for p in checked_paths)
    st.error(
        "Data file not found.\n"
        "Make sure `owid-covid-data.csv` exists in `mnt/data` or `/mnt/data`.\n\n"
        f"Checked paths:\n{checked}"
    )
    st.stop()


df = load_data(data_path)
min_date = df["date"].min().date()
DATE_CAP = pd.to_datetime("2024-08-01").date()
max_date = min(df["date"].max().date(), DATE_CAP)
all_continents = sorted(df["continent"].dropna().unique().tolist())

st.markdown(
    f"""
    <div class="hero">
        <h1>Interactive COVID-19 Dashboard</h1>
        <p>
            Inspired by Johns Hopkins style with richer filters, flexible date ranges,
            country-level comparisons, and an interactive global map powered by Our World in Data.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.header("Controls")
    date_range = st.date_input(
        "Select date range",
        value=(max(min_date, pd.to_datetime("2020-03-01").date()), max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
    else:
        start_date = pd.to_datetime(min_date)
        end_date = pd.to_datetime(max_date)

    continents = st.multiselect(
        "Continents",
        options=all_continents,
        default=all_continents,
    )

    locations_options = available_locations(df, tuple(continents))
    default_locations = [loc for loc in ["United States", "India", "Brazil", "United Kingdom"] if loc in locations_options]
    selected_locations = st.multiselect(
        "Countries/Regions",
        options=locations_options,
        default=default_locations,
    )

    metric = st.selectbox(
        "Primary metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda x: METRIC_LABELS.get(x, x),
        index=0,
    )

    compare_mode = st.radio(
        "Comparison mode",
        options=["Absolute values", "Per million (when available)"],
        index=0,
    )

    top_n = st.slider("Number of countries in ranking", min_value=5, max_value=25, value=10)
    show_only_with_data = st.toggle("Hide rows with missing values", value=True)

filtered = df[(df["continent"].isin(continents)) & (df["date"].between(start_date, end_date))].copy()
if selected_locations:
    filtered = filtered[filtered["location"].isin(selected_locations)]

if filtered.empty:
    st.warning("No data found for the selected filters.")
    st.stop()

metric_for_map = metric
if compare_mode != "Absolute values":
    candidate = f"{metric}_per_million"
    if candidate in df.columns:
        metric_for_map = candidate

latest_global = get_metric_snapshot(df[df["continent"].isin(continents)], start_date, end_date, metric)
latest_global_full = get_latest_in_range(df[df["continent"].isin(continents)], start_date, end_date)
if selected_locations:
    latest_global = latest_global[latest_global["location"].isin(selected_locations)]
    latest_global_full = latest_global_full[latest_global_full["location"].isin(selected_locations)]

if show_only_with_data:
    latest_global = latest_global.dropna(subset=[metric])

base_for_global = df[(df["continent"].isin(continents)) & (df["date"].between(start_date, end_date))].copy()
if selected_locations:
    base_for_global = base_for_global[base_for_global["location"].isin(selected_locations)]

latest_date_in_scope = base_for_global["date"].max()
latest_snapshot = base_for_global[base_for_global["date"] == latest_date_in_scope].copy()

world_scope = df[(df["location"] == "World") & (df["date"].between(start_date, end_date))].copy()
world_latest_date = world_scope["date"].max() if not world_scope.empty else pd.NaT
world_latest_row = world_scope[world_scope["date"] == world_latest_date].tail(1) if not pd.isna(world_latest_date) else pd.DataFrame()
world_prev_row = (
    world_scope[world_scope["date"] == (world_latest_date - pd.Timedelta(days=1))].tail(1)
    if not pd.isna(world_latest_date)
    else pd.DataFrame()
)

if not world_latest_row.empty:
    k1 = float(world_latest_row["total_cases"].iloc[0]) if pd.notna(world_latest_row["total_cases"].iloc[0]) else 0.0
    k2 = float(world_latest_row["total_deaths"].iloc[0]) if pd.notna(world_latest_row["total_deaths"].iloc[0]) else 0.0
    k3 = float(world_latest_row["people_fully_vaccinated"].iloc[0]) if pd.notna(world_latest_row["people_fully_vaccinated"].iloc[0]) else 0.0
    k4 = max(k1 - k2, 0.0)
else:
    k1 = latest_snapshot["total_cases"].fillna(0).sum()
    k2 = latest_snapshot["total_deaths"].fillna(0).sum()
    k3 = latest_snapshot["people_fully_vaccinated"].fillna(0).sum()
    k4 = (latest_snapshot["total_cases"].fillna(0) - latest_snapshot["total_deaths"].fillna(0)).clip(lower=0).sum()

prev_day = latest_date_in_scope - pd.Timedelta(days=1)
prev_snapshot = base_for_global[base_for_global["date"] == prev_day].copy()

def get_delta(col: str):
    if not world_latest_row.empty and not world_prev_row.empty:
        latest_val = world_latest_row[col].iloc[0] if col in world_latest_row.columns else np.nan
        prev_val = world_prev_row[col].iloc[0] if col in world_prev_row.columns else np.nan
        latest_val = float(latest_val) if pd.notna(latest_val) else 0.0
        prev_val = float(prev_val) if pd.notna(prev_val) else 0.0
        return latest_val - prev_val
    if prev_snapshot.empty:
        return None
    return latest_snapshot[col].fillna(0).sum() - prev_snapshot[col].fillna(0).sum()


def get_recovered_delta():
    if not world_latest_row.empty and not world_prev_row.empty:
        latest_cases = world_latest_row["total_cases"].iloc[0] if "total_cases" in world_latest_row.columns else np.nan
        latest_deaths = world_latest_row["total_deaths"].iloc[0] if "total_deaths" in world_latest_row.columns else np.nan
        prev_cases = world_prev_row["total_cases"].iloc[0] if "total_cases" in world_prev_row.columns else np.nan
        prev_deaths = world_prev_row["total_deaths"].iloc[0] if "total_deaths" in world_prev_row.columns else np.nan
        latest_recovered = max((float(latest_cases) if pd.notna(latest_cases) else 0.0) - (float(latest_deaths) if pd.notna(latest_deaths) else 0.0), 0.0)
        prev_recovered = max((float(prev_cases) if pd.notna(prev_cases) else 0.0) - (float(prev_deaths) if pd.notna(prev_deaths) else 0.0), 0.0)
        return latest_recovered - prev_recovered
    if prev_snapshot.empty:
        return None
    latest_recovered = (latest_snapshot["total_cases"].fillna(0) - latest_snapshot["total_deaths"].fillna(0)).clip(lower=0).sum()
    prev_recovered = (prev_snapshot["total_cases"].fillna(0) - prev_snapshot["total_deaths"].fillna(0)).clip(lower=0).sum()
    return latest_recovered - prev_recovered

c1, c2, c3, c4 = st.columns(4)
for col, label, value, delta in [
    (c1, "Total Cases", k1, get_delta("total_cases")),
    (c2, "Total Deaths", k2, get_delta("total_deaths")),
    (c3, "Fully Vaccinated", k3, get_delta("people_fully_vaccinated")),
    (c4, "Total Recovered", k4, get_recovered_delta()),
]:
    with col:
        st.markdown(
            f"""
            <div class="kpi-card">
                <div class="kpi-label">{label}</div>
                <div class="kpi-value">{fmt_num(value)}</div>
                <div class="kpi-sub">Daily change: {fmt_num(delta) if delta is not None else '-'}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

left, right = st.columns([2.3, 1])

with left:
    st.markdown('<div class="panel-title">Global Map</div>', unsafe_allow_html=True)
    map_df = get_metric_snapshot(df[df["continent"].isin(continents)], start_date, end_date, metric_for_map)
    if selected_locations:
        map_df = map_df[map_df["location"].isin(selected_locations)]
    if show_only_with_data:
        map_df = map_df.dropna(subset=[metric_for_map])

    if map_df.empty:
        st.info("Not enough data to render the map.")
    else:
        # Use unique helper columns for hover custom_data to avoid duplicate-name errors
        # when metric_for_map matches one of the summary fields (e.g., total_cases).
        map_plot_df = map_df.assign(
            __metric_value=map_df[metric_for_map],
            __total_cases=map_df["total_cases"],
            __total_deaths=map_df["total_deaths"],
            __fully_vaccinated=map_df["people_fully_vaccinated"],
        )
        fig_map = px.scatter_geo(
            map_plot_df,
            locations="iso_code",
            hover_name="location",
            size=metric_for_map,
            color=metric_for_map,
            projection="natural earth",
            custom_data=["location", "__metric_value", "__total_cases", "__total_deaths", "__fully_vaccinated"],
        )
        fig_map.update_traces(
            marker=dict(sizemode="area", sizeref=max(map_df[metric_for_map].max() / 60**2, 1e-9), line=dict(width=0.4, color="#ffb3a7")),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                + f"{METRIC_LABELS.get(metric_for_map, metric_for_map)}: " + "%{customdata[1]:,.0f}<br>"
                + "Total Cases: %{customdata[2]:,.0f}<br>"
                + "Total Deaths: %{customdata[3]:,.0f}<br>"
                + "Fully Vaccinated: %{customdata[4]:,.0f}<extra></extra>"
            ),
        )
        fig_map.update_layout(
            height=560,
            margin=dict(l=0, r=0, t=0, b=0),
            paper_bgcolor="#09111e",
            plot_bgcolor="#09111e",
            font=dict(color="#ecf2ff"),
            geo=dict(bgcolor="#09111e", showland=True, landcolor="#2e3138", oceancolor="#02111f", showocean=True),
            coloraxis_colorbar=dict(title=METRIC_LABELS.get(metric_for_map, metric_for_map)),
        )
        st.plotly_chart(fig_map, width="stretch")

with right:
    st.markdown('<div class="panel-title">Top Countries by Metric</div>', unsafe_allow_html=True)
    ranking_metric = metric
    ranking_label = METRIC_LABELS.get(ranking_metric, ranking_metric)
    rank_base = latest_global.assign(
        __rank_value=latest_global[ranking_metric],
        __total_cases=latest_global["total_cases"],
        __total_deaths=latest_global["total_deaths"],
        __fully_vaccinated=latest_global["people_fully_vaccinated"],
    )
    rank_df = rank_base[["location", "continent", "__rank_value", "__total_cases", "__total_deaths", "__fully_vaccinated"]].copy()
    rank_df = rank_df.dropna(subset=["__rank_value"]).sort_values("__rank_value", ascending=False).head(top_n)
    rank_df = rank_df.rename(columns={
        "location": "Country",
        "continent": "Continent",
        "__rank_value": f"{ranking_label} (Selected Metric)",
        "__total_cases": "Total Cases",
        "__total_deaths": "Total Deaths",
        "__fully_vaccinated": "Fully Vaccinated",
    })
    st.dataframe(rank_df, width="stretch", height=560)

st.markdown("### Time Trends")

series_cols = st.columns([1.1, 1.1, 0.8])
with series_cols[0]:
    series_metric = st.selectbox(
        "Time-series metric",
        options=list(METRIC_LABELS.keys()),
        format_func=lambda x: METRIC_LABELS.get(x, x),
        index=0,
        key="series_metric",
    )
with series_cols[1]:
    smooth_window = st.slider("Moving average window", 1, 30, 7)
with series_cols[2]:
    chart_type = st.selectbox("Chart type", ["Line", "Area", "Bar"])

plot_df = filtered[["date", "location", series_metric]].copy().dropna()
if plot_df.empty:
    st.info("No time-series data found under the current filters.")
else:
    plot_df = plot_df.sort_values(["location", "date"])
    plot_df["value_smooth"] = plot_df.groupby("location")[series_metric].transform(lambda s: s.rolling(smooth_window, min_periods=1).mean())

    if chart_type == "Line":
        fig_series = px.line(plot_df, x="date", y="value_smooth", color="location")
    elif chart_type == "Area":
        fig_series = px.area(plot_df, x="date", y="value_smooth", color="location")
    else:
        fig_series = px.bar(plot_df, x="date", y="value_smooth", color="location")

    fig_series.update_layout(
        height=380,
        paper_bgcolor="#09111e",
        plot_bgcolor="#09111e",
        font=dict(color="#ecf2ff"),
        margin=dict(l=10, r=10, t=10, b=10),
        legend_title_text="Country",
        yaxis_title=METRIC_LABELS.get(series_metric, series_metric),
        xaxis_title="Date",
    )
    st.plotly_chart(fig_series, width="stretch")

st.markdown("### Start vs End Comparison")
first_vals, last_vals = get_first_last_values(df[df["continent"].isin(continents)], start_date, end_date, metric, tuple(selected_locations))
compare_df = first_vals.merge(last_vals, on="location", how="outer")
compare_df["change"] = compare_df["end_value"].fillna(0) - compare_df["start_value"].fillna(0)
compare_df = compare_df.sort_values("change", ascending=False)
compare_df = compare_df.rename(columns={
    "location": "Country",
    "start_value": "Start Value",
    "end_value": "End Value",
    "change": "Net Change",
})
st.dataframe(compare_df, width="stretch", height=320)

bottom_left, bottom_right = st.columns([1.25, 1])

with bottom_left:
    st.markdown("### Correlation Analysis")
    x_metric = st.selectbox(
        "X Axis",
        options=["total_cases_per_million", "total_deaths_per_million", "gdp_per_capita", "median_age", "hospital_beds_per_thousand", "positive_rate", "stringency_index"],
        index=0,
        format_func=lambda x: METRIC_LABELS.get(x, x.replace("_", " ").title()),
    )
    y_metric = st.selectbox(
        "Y Axis",
        options=["total_deaths_per_million", "people_fully_vaccinated_per_hundred", "positive_rate", "reproduction_rate", "life_expectancy"],
        index=0,
        format_func=lambda x: METRIC_LABELS.get(x, x.replace("_", " ").title()),
    )
    scatter_df = latest_global_full[["location", "continent", x_metric, y_metric, "population"]].dropna()
    if not scatter_df.empty:
        fig_scatter = px.scatter(
            scatter_df,
            x=x_metric,
            y=y_metric,
            color="continent",
            size="population",
            hover_name="location",
            size_max=45,
        )
        fig_scatter.update_layout(
            height=420,
            paper_bgcolor="#09111e",
            plot_bgcolor="#09111e",
            font=dict(color="#ecf2ff"),
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig_scatter, width="stretch")
    else:
        st.info("Not enough data for correlation analysis.")

with bottom_right:
    st.markdown("### Country Detail Snapshot")
    profile_cols = [
        "location", "continent", "population", "total_cases", "total_deaths",
        "people_vaccinated", "people_fully_vaccinated", "total_tests",
        "positive_rate", "reproduction_rate", "stringency_index", "life_expectancy"
    ]
    profile_df = latest_global_full[profile_cols].copy().sort_values("total_cases", ascending=False)
    profile_df = profile_df.rename(columns={
        "location": "Country",
        "continent": "Continent",
        "population": "Population",
        "total_cases": "Total Cases",
        "total_deaths": "Total Deaths",
        "people_vaccinated": "At Least One Dose",
        "people_fully_vaccinated": "Fully Vaccinated",
        "total_tests": "Total Tests",
        "positive_rate": "Positive Rate",
        "reproduction_rate": "Reproduction Rate",
        "stringency_index": "Stringency Index",
        "life_expectancy": "Life Expectancy",
    })
    st.dataframe(profile_df, width="stretch", height=420)

st.caption(
    f"Latest date in current scope: {latest_date_in_scope.date()} | KPI source: World row (OWID) | Source: Our World in Data | File: {data_path.name}"
)









