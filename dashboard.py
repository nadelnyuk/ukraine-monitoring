"""
dashboard.py — Streamlit dashboard for politician and keyword mention monitoring.

Tabs:
  1. Politician Mention Monitoring  — reads data/mentions.csv
  2. Keywords Mention Monitoring    — reads data/topic_mentions.csv

Run locally:
    streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ukraine Monitoring",
    page_icon="📊",
    layout="wide",
)

# ─── Ukrainian → English name mapping ─────────────────────────────────────────

NAME_MAP: dict[str, str] = {
    "Іллєнко Андрій": "Illienko Andriy",
    "Арахамія Давид": "Arakhamia David",
    "Арестович Олексій": "Arestovych Oleksiy",
    "Безугла Мар'яна": "Bezuhla Mariana",
    "Береза Борислав": "Bereza Borislav",
    "Берлінська Марія": "Berlinska Maria",
    "Бойко Юрій": "Boiko Yuriy",
    "Бровді Роберт": "Brovdi Robert",
    "Буданов Кирило": "Budanov Kyrylo",
    "Бутусов Юрій": "Butusov Yuriy",
    "Білецький Андрій": "Biletsky Andriy",
    "Верещук Ірина": "Vereshchuk Iryna",
    "Вілкул Олександр": "Vilkul Oleksandr",
    "Гетманцев Данило": "Hetmantsev Danylo",
    "Гордон Дмитро": "Gordon Dmytro",
    "Железняк Ярослав": "Zheleznyak Yaroslav",
    "Залужний Валерій": "Zaluzhnyi Valerii",
    "Зеленський Володимир": "Zelensky Volodymyr",
    "Карась Євген": "Karas Yevhen",
    "Кличко Віталій": "Klitschko Vitali",
    "Кривонос Сергій": "Kryvonos Serhiy",
    "Кротевич Богдан": "Krotevych Bohdan",
    "Кулеба Дмитро": "Kuleba Dmytro",
    "Кім Віталій": "Kim Vitaliy",
    "Луценко Юрій": "Lutsenko Yuriy",
    "Ляшко Олег": "Lyashko Oleh",
    "Малюк Василь": "Maliuk Vasyl",
    "Маркус Валерій": "Markus Valerii",
    "Мураєв Євген": "Muraiev Yevhen",
    "Подоляк Михайло": "Podolyak Mykhailo",
    "Порошенко Петро": "Poroshenko Petro",
    "Портніков Віталій": "Portnikov Vitaliy",
    "Притула Сергій": "Prytula Serhiy",
    "Прокопенко Денис": "Prokopenko Denys",
    "Разумков Дмитро": "Razumkov Dmytro",
    "Свириденко Юлія": "Svyrydenko Yulia",
    "Стерненко Сергій": "Sternenko Serhiy",
    "Стефанчук Руслан": "Stefanchuk Ruslan",
    "Терехов Ігор": "Terekhov Ihor",
    "Тимошенко Юлія": "Tymoshenko Yulia",
    "Умеров Рустем": "Umerov Rustem",
    "Усик Олександр": "Usyk Oleksandr",
    "Чмут Тарас": "Chmut Taras",
    "Яценюк Арсеній": "Yatseniuk Arseniy",
}

NAME_MAP_REVERSE: dict[str, str] = {v: k for k, v in NAME_MAP.items()}


def to_en(name: str) -> str:
    """Translate a Ukrainian politician name to English; fall back to the original."""
    return NAME_MAP.get(name, name)


# ─── Data loading ─────────────────────────────────────────────────────────────

MENTIONS_FILE = Path(__file__).parent / "data" / "mentions.csv"
TOPIC_FILE = Path(__file__).parent / "data" / "topic_mentions.csv"


@st.cache_data
def load_mentions() -> pd.DataFrame:
    """Load and validate mentions.csv.

    Returns:
        DataFrame with columns: date (datetime), politician (str), mentions (int).
    """
    if not MENTIONS_FILE.exists():
        st.error(
            f"Data file not found: `{MENTIONS_FILE}`\n\n"
            "Run `python process_mentions.py` first."
        )
        st.stop()

    df = pd.read_csv(MENTIONS_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date", "politician", "mentions"])
    df["mentions"] = df["mentions"].astype(int)
    df = df.sort_values("date")
    return df


@st.cache_data
def load_topic_mentions() -> pd.DataFrame:
    """Load and validate topic_mentions.csv.

    Returns:
        DataFrame with columns: date, username, category_id, category,
        subcategory_id, subcategory, post_count.
    """
    if not TOPIC_FILE.exists():
        st.error(
            f"Data file not found: `{TOPIC_FILE}`\n\n"
            "Run `python process_topic_mentions.py` first."
        )
        st.stop()

    df = pd.read_csv(TOPIC_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date", "username", "category", "subcategory", "post_count"])
    df["post_count"] = df["post_count"].astype(int)
    df = df.sort_values("date")
    return df


# ─── Shared layout helpers ────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    margin=dict(t=40, b=40),
    height=420,
)


def _channel_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render a channel dropdown above the caller's chart and return filtered df.

    Args:
        df: DataFrame that must have a ``username`` column.
        key: Unique Streamlit widget key (to avoid duplicate-key errors).

    Returns:
        The same DataFrame filtered to the selected channel, or unfiltered if
        "All channels" is selected.
    """
    channels = ["All channels"] + sorted(df["username"].unique().tolist())
    selected = st.selectbox("Channel filter", channels, key=key)
    if selected != "All channels":
        df = df[df["username"] == selected]
    return df


# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab1, tab2 = st.tabs(
    ["📰 Politician Mention Monitoring", "🔍 Keywords Mention Monitoring"]
)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — Politician Mention Monitoring (unchanged logic)
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    st.title("📊 Politician Mention Monitoring")

    df_raw = load_mentions()

    min_date = df_raw["date"].min().date()
    max_date = df_raw["date"].max().date()

    date_from, date_to = st.date_input(
        "Date range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        key="pol_date_range",
    )

    st.caption(
        f"Telegram channels · {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}"
    )

    mask = (df_raw["date"].dt.date >= date_from) & (df_raw["date"].dt.date <= date_to)
    df = df_raw[mask].copy()

    if df.empty:
        st.warning("No data for the selected date range.")
        st.stop()

    def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot to wide format with one column per politician, zeros for missing days.

        Args:
            df: Filtered mentions DataFrame (date, politician, mentions columns).

        Returns:
            Wide DataFrame indexed by date, with a complete daily date range.
        """
        pivot = df.pivot_table(
            index="date",
            columns="politician",
            values="mentions",
            aggfunc="sum",
            fill_value=0,
        )
        full_range = pd.date_range(pivot.index.min(), pivot.index.max(), freq="D")
        pivot = pivot.reindex(full_range, fill_value=0)
        pivot.index.name = "date"
        pivot.columns = [to_en(col) for col in pivot.columns]
        return pivot

    pivot = build_pivot(df)

    # Chart 1: Top-5 by total mentions
    st.markdown("---")
    st.subheader("Top 5 by Total Mentions")

    totals = pivot.sum().sort_values(ascending=False)
    top5_names = totals.head(5).index.tolist()

    top5_data = pivot[top5_names].reset_index().melt(
        id_vars="date",
        var_name="Politician",
        value_name="Mentions",
    )

    fig1 = px.line(
        top5_data,
        x="date",
        y="Mentions",
        color="Politician",
        labels={"date": "Date", "Mentions": "Mention count"},
        template="plotly_white",
    )
    fig1.update_traces(line_width=2)
    fig1.update_layout(**_CHART_LAYOUT)
    st.plotly_chart(fig1, use_container_width=True)

    col_left, _ = st.columns([2, 3])
    with col_left:
        st.caption("Total mentions in selected range:")
        totals_df = (
            totals.reset_index()
            .rename(columns={"politician": "Politician", 0: "Total"})
            .head(10)
        )
        st.dataframe(totals_df, use_container_width=True, hide_index=True)

    # Chart 2: Head-to-head comparison
    st.markdown("---")
    st.subheader("Head-to-Head Comparison")

    all_politicians = sorted(pivot.columns.tolist())
    zelensky_en = to_en("Зеленський Володимир")

    col1, col2 = st.columns(2)
    with col1:
        pol_a = st.selectbox(
            "First politician",
            options=all_politicians,
            index=all_politicians.index(zelensky_en)
            if zelensky_en in all_politicians
            else 0,
        )
    with col2:
        default_b_index = next(
            (i for i, p in enumerate(all_politicians) if p != pol_a), 0
        )
        pol_b = st.selectbox(
            "Second politician",
            options=all_politicians,
            index=default_b_index,
        )

    if pol_a == pol_b:
        st.info("Select two different politicians to compare.")
    else:
        compare_data = pivot[[pol_a, pol_b]].reset_index().melt(
            id_vars="date",
            var_name="Politician",
            value_name="Mentions",
        )

        fig2 = px.line(
            compare_data,
            x="date",
            y="Mentions",
            color="Politician",
            labels={"date": "Date", "Mentions": "Mention count"},
            template="plotly_white",
            color_discrete_sequence=["#1f77b4", "#d62728"],
        )
        fig2.update_traces(line_width=2.5)
        fig2.update_layout(**_CHART_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

        total_a = int(pivot[pol_a].sum())
        total_b = int(pivot[pol_b].sum())
        peak_a = pivot[pol_a].idxmax()
        peak_b = pivot[pol_b].idxmax()

        c1, c2 = st.columns(2)
        with c1:
            st.metric(label=f"Total: {pol_a}", value=f"{total_a:,}")
            st.caption(
                f"Peak: {peak_a.strftime('%d.%m.%Y')} ({int(pivot[pol_a].max()):,} mentions)"
            )
        with c2:
            st.metric(label=f"Total: {pol_b}", value=f"{total_b:,}")
            st.caption(
                f"Peak: {peak_b.strftime('%d.%m.%Y')} ({int(pivot[pol_b].max()):,} mentions)"
            )


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — Keywords Mention Monitoring
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.title("🔍 Keywords Mention Monitoring")

    tm = load_topic_mentions()

    # ── Chart 1: Mentions by category over time ────────────────────────────────

    st.markdown("---")
    st.subheader("Mentions by Category Over Time")

    tm_cat = _channel_filter(tm, key="cat_channel")

    cat_daily = (
        tm_cat.groupby(["date", "category"], as_index=False)["post_count"]
        .sum()
        .sort_values("date")
    )

    fig_cat = px.line(
        cat_daily,
        x="date",
        y="post_count",
        color="category",
        labels={"date": "Date", "post_count": "Post count", "category": "Category"},
        template="plotly_white",
    )
    fig_cat.update_traces(line_width=2)
    fig_cat.update_layout(**_CHART_LAYOUT)
    st.plotly_chart(fig_cat, use_container_width=True)

    # ── Chart 2: Most active channels (bar chart) ──────────────────────────────

    st.markdown("---")
    st.subheader("Most Mentioned Telegram Channels")

    channel_totals = (
        tm.groupby("username", as_index=False)["post_count"]
        .sum()
        .sort_values("post_count", ascending=False)
    )

    fig_bar = px.bar(
        channel_totals,
        x="username",
        y="post_count",
        labels={"username": "Channel", "post_count": "Total posts"},
        template="plotly_white",
        color_discrete_sequence=["#2196F3"],
    )
    fig_bar.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=80),
        height=420,
        xaxis_tickangle=-45,
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart 3: Subcategories within a selected category ─────────────────────

    st.markdown("---")
    st.subheader("Subcategory Breakdown by Category")

    # Two controls side-by-side: category selector + channel filter
    ctrl1, ctrl2 = st.columns([2, 3])
    with ctrl1:
        all_categories = sorted(tm["category"].unique().tolist())
        chosen_cat = st.selectbox("Category", all_categories, key="subcat_category")
    with ctrl2:
        tm_subcat_base = tm[tm["category"] == chosen_cat]
        tm_subcat = _channel_filter(tm_subcat_base, key="subcat_channel")

    subcat_daily = (
        tm_subcat.groupby(["date", "subcategory"], as_index=False)["post_count"]
        .sum()
        .sort_values("date")
    )

    if subcat_daily.empty:
        st.info("No data for the selected category / channel combination.")
    else:
        fig_subcat = px.line(
            subcat_daily,
            x="date",
            y="post_count",
            color="subcategory",
            labels={
                "date": "Date",
                "post_count": "Post count",
                "subcategory": "Subcategory",
            },
            template="plotly_white",
        )
        fig_subcat.update_traces(line_width=2)
        fig_subcat.update_layout(**_CHART_LAYOUT)
        st.plotly_chart(fig_subcat, use_container_width=True)

    # ── Chart 4: Top-5 subcategories overall ──────────────────────────────────

    st.markdown("---")
    st.subheader("Top 5 Subcategories Overall")

    tm_top5 = _channel_filter(tm, key="top5_channel")

    top5_subcats = (
        tm_top5.groupby("subcategory")["post_count"]
        .sum()
        .sort_values(ascending=False)
        .head(5)
        .index.tolist()
    )

    top5_daily = (
        tm_top5[tm_top5["subcategory"].isin(top5_subcats)]
        .groupby(["date", "subcategory"], as_index=False)["post_count"]
        .sum()
        .sort_values("date")
    )

    fig_top5 = px.line(
        top5_daily,
        x="date",
        y="post_count",
        color="subcategory",
        labels={
            "date": "Date",
            "post_count": "Post count",
            "subcategory": "Subcategory",
        },
        template="plotly_white",
    )
    fig_top5.update_traces(line_width=2)
    fig_top5.update_layout(**_CHART_LAYOUT)
    st.plotly_chart(fig_top5, use_container_width=True)

    # ── Chart 5: Heatmap — top-10 channels × categories ───────────────────────

    st.markdown("---")
    st.subheader("Top 10 Channels × Categories Heatmap")

    top10_channels = (
        tm.groupby("username")["post_count"]
        .sum()
        .sort_values(ascending=False)
        .head(10)
        .index.tolist()
    )

    heatmap_df = (
        tm[tm["username"].isin(top10_channels)]
        .groupby(["username", "category"], as_index=False)["post_count"]
        .sum()
    )

    # Pivot to matrix: rows = channels (sorted by total desc), cols = categories
    heatmap_pivot = heatmap_df.pivot_table(
        index="username",
        columns="category",
        values="post_count",
        fill_value=0,
    )
    # Keep channels in descending-total order on the y-axis
    heatmap_pivot = heatmap_pivot.reindex(top10_channels)

    fig_heat = go.Figure(
        data=go.Heatmap(
            z=heatmap_pivot.values,
            x=heatmap_pivot.columns.tolist(),
            y=heatmap_pivot.index.tolist(),
            colorscale="Blues",
            hoverongaps=False,
            hovertemplate="Channel: %{y}<br>Category: %{x}<br>Posts: %{z}<extra></extra>",
        )
    )
    fig_heat.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=100, l=160),
        height=480,
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),  # highest total at top
    )
    st.plotly_chart(fig_heat, use_container_width=True)
