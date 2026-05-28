"""
dashboard.py — Streamlit dashboard for Ukraine Telegram monitoring.

Tabs:
  1. About              — methodology, dataset stats
  2. Politician Mentions — reads mentions.csv + mentions_by_type.csv
  3. Keywords           — reads topic_mentions.csv + shows keyword definitions
  4. Audience Insights  — reads audience_insights.csv

Run locally:
    streamlit run dashboard.py
"""

from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from keyword_taxonomy import load_taxonomy

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Ukraine Monitoring",
    page_icon="📊",
    layout="wide",
)

# ─── Methodology text ─────────────────────────────────────────────────────────
# Boss will provide this text. Paste it between the triple-quotes below.

METHODOLOGY_TEXT: str = ""

# ─── Ukrainian → English name mapping ────────────────────────────────────────

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
    """Translate a Ukrainian politician name to English; fall back to original."""
    return NAME_MAP.get(name, name)


# ─── File paths ───────────────────────────────────────────────────────────────

_HERE = Path(__file__).parent

MENTIONS_FILE = _HERE / "data" / "mentions.csv"
MENTIONS_BY_TYPE_FILE = _HERE / "data" / "mentions_by_type.csv"
TOPIC_FILE = _HERE / "data" / "topic_mentions.csv"
AUDIENCE_FILE = _HERE / "data" / "audience_insights.csv"
CHANNEL_TYPES_FILE = _HERE / "data" / "channel_types.csv"
TAXONOMY_FILE = _HERE / "Monitoring Keywords by Topic.md"


# ─── Data loading ─────────────────────────────────────────────────────────────

@st.cache_data
def load_mentions() -> pd.DataFrame:
    """Load mentions.csv → date (datetime), politician (str), mentions (int)."""
    if not MENTIONS_FILE.exists():
        st.error(f"Data file not found: `{MENTIONS_FILE}`\n\nRun `python process_mentions.py` first.")
        st.stop()
    df = pd.read_csv(MENTIONS_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date", "politician", "mentions"])
    df["mentions"] = df["mentions"].astype(int)
    return df.sort_values("date")


@st.cache_data
def load_mentions_by_type() -> pd.DataFrame | None:
    """Load mentions_by_type.csv if it exists; return None otherwise."""
    if not MENTIONS_BY_TYPE_FILE.exists():
        return None
    df = pd.read_csv(MENTIONS_BY_TYPE_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date", "politician", "channel_type", "mentions"])
    df["mentions"] = df["mentions"].astype(int)
    return df.sort_values("date")


@st.cache_data
def load_topic_mentions() -> pd.DataFrame:
    """Load topic_mentions.csv."""
    if not TOPIC_FILE.exists():
        st.error(f"Data file not found: `{TOPIC_FILE}`\n\nRun `python process_topic_mentions.py` first.")
        st.stop()
    df = pd.read_csv(TOPIC_FILE, parse_dates=["date"])
    df = df.dropna(subset=["date", "username", "category", "subcategory", "post_count"])
    df["post_count"] = df["post_count"].astype(int)
    return df.sort_values("date")


@st.cache_data
def load_audience_insights() -> pd.DataFrame | None:
    """Load audience_insights.csv if it exists; return None otherwise."""
    if not AUDIENCE_FILE.exists():
        return None
    df = pd.read_csv(AUDIENCE_FILE)
    df = df.dropna(subset=["username", "total_posts", "total_views"])
    for col in df.columns:
        if col != "username":
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)
    return df


@st.cache_data
def load_channel_types() -> pd.DataFrame:
    """Load channel_types.csv, or return an empty DataFrame if missing."""
    if not CHANNEL_TYPES_FILE.exists():
        return pd.DataFrame(columns=["username", "channel_type", "notes"])
    df = pd.read_csv(CHANNEL_TYPES_FILE, dtype=str).fillna("")
    return df[["username", "channel_type"]].drop_duplicates("username")


@st.cache_data
def load_keyword_taxonomy() -> list[dict]:
    """Load and return the taxonomy entries (cached for the session)."""
    return load_taxonomy(TAXONOMY_FILE)


# ─── Shared layout helpers ────────────────────────────────────────────────────

_CHART_LAYOUT = dict(
    template="plotly_white",
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    hovermode="x unified",
    margin=dict(t=40, b=40),
    height=420,
)


def _channel_filter(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Render a channel dropdown and return filtered df (topic mentions tab)."""
    channels = ["All channels"] + sorted(df["username"].unique().tolist())
    selected = st.selectbox("Channel filter", channels, key=key)
    if selected != "All channels":
        df = df[df["username"] == selected]
    return df


# ─── Category definitions for Audience Insights ──────────────────────────────

CATEGORY_COLS = ["political", "electoral", "military", "economy", "religion", "international"]

CATEGORY_LABELS = {
    "political": "Political",
    "electoral": "Electoral",
    "military": "Military",
    "economy": "Economy",
    "religion": "Religion & Language",
    "international": "International",
}

# ─── Tabs ─────────────────────────────────────────────────────────────────────

tab_about, tab_politicians, tab_keywords, tab_audience = st.tabs([
    "ℹ️ About",
    "📰 Politician Mentions",
    "🔍 Keywords",
    "📡 Audience Insights",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB — About
# ══════════════════════════════════════════════════════════════════════════════

with tab_about:
    st.title("Ukraine Telegram Monitoring Dashboard")

    if METHODOLOGY_TEXT:
        st.markdown(METHODOLOGY_TEXT)
    else:
        st.info("Methodology paragraph coming soon.")

    st.markdown("---")
    st.subheader("Dataset Overview")

    _df_raw = load_mentions()
    _tm = load_topic_mentions()

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Telegram Channels Tracked", f"{_tm['username'].nunique():,}")
    with col2:
        kw_posts = int(_tm["post_count"].sum())
        st.metric("Keyword-Matching Posts", f"{kw_posts:,}")
    with col3:
        date_start = _df_raw["date"].min().strftime("%b %d, %Y")
        date_end = _df_raw["date"].max().strftime("%b %d, %Y")
        st.metric("Date Range", f"{date_start} — {date_end}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Politician Mentions
# ══════════════════════════════════════════════════════════════════════════════

with tab_politicians:
    st.title("📊 Politician Mention Monitoring")

    df_raw = load_mentions()
    df_by_type = load_mentions_by_type()
    ct = load_channel_types()

    min_date = df_raw["date"].min().date()
    max_date = df_raw["date"].max().date()

    # ── Controls ──────────────────────────────────────────────────────────────
    ctrl_col1, ctrl_col2 = st.columns([3, 2])

    with ctrl_col1:
        date_from, date_to = st.date_input(
            "Date range",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
            key="pol_date_range",
        )

    with ctrl_col2:
        if df_by_type is not None and not ct.empty:
            type_options = ["All types"] + sorted(
                df_by_type["channel_type"].unique().tolist()
            )
            selected_type = st.selectbox(
                "Channel type filter",
                type_options,
                key="pol_channel_type",
            )
        else:
            selected_type = "All types"
            if df_by_type is None:
                st.caption(
                    "Channel type filter not available yet.  \n"
                    "Run `python process_mentions.py --full` to enable it."
                )

    st.caption(
        f"Telegram channels · {date_from.strftime('%d.%m.%Y')} — {date_to.strftime('%d.%m.%Y')}"
        + (f" · {selected_type}" if selected_type != "All types" else "")
    )

    # ── Build working dataframe ───────────────────────────────────────────────
    if selected_type != "All types" and df_by_type is not None:
        mask = (
            (df_by_type["date"].dt.date >= date_from)
            & (df_by_type["date"].dt.date <= date_to)
            & (df_by_type["channel_type"] == selected_type)
        )
        df = df_by_type[mask].groupby(["date", "politician"], as_index=False)["mentions"].sum()
    else:
        mask = (df_raw["date"].dt.date >= date_from) & (df_raw["date"].dt.date <= date_to)
        df = df_raw[mask].copy()

    if df.empty:
        st.warning("No data for the selected date range / channel type.")
        st.stop()

    def build_pivot(df: pd.DataFrame) -> pd.DataFrame:
        """Pivot to wide format indexed by date, English politician names."""
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

    # ── Chart 1: Top-5 by Total Mentions ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Top 5 by Total Mentions")

    totals = pivot.sum().sort_values(ascending=False)
    top5_names = totals.head(5).index.tolist()

    top5_data = pivot[top5_names].reset_index().melt(
        id_vars="date", var_name="Politician", value_name="Mentions"
    )

    fig1 = px.line(
        top5_data,
        x="date", y="Mentions", color="Politician",
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

    # ── Chart 2: Head-to-Head Comparison ─────────────────────────────────────
    st.markdown("---")
    st.subheader("Head-to-Head Comparison")

    all_politicians = sorted(pivot.columns.tolist())
    zelensky_en = to_en("Зеленський Володимир")

    col1, col2 = st.columns(2)
    with col1:
        pol_a = st.selectbox(
            "First politician",
            options=all_politicians,
            index=all_politicians.index(zelensky_en) if zelensky_en in all_politicians else 0,
        )
    with col2:
        default_b_index = next((i for i, p in enumerate(all_politicians) if p != pol_a), 0)
        pol_b = st.selectbox(
            "Second politician",
            options=all_politicians,
            index=default_b_index,
        )

    if pol_a == pol_b:
        st.info("Select two different politicians to compare.")
    else:
        compare_data = pivot[[pol_a, pol_b]].reset_index().melt(
            id_vars="date", var_name="Politician", value_name="Mentions"
        )

        fig2 = px.line(
            compare_data,
            x="date", y="Mentions", color="Politician",
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
            st.caption(f"Peak: {peak_a.strftime('%d.%m.%Y')} ({int(pivot[pol_a].max()):,} mentions)")
        with c2:
            st.metric(label=f"Total: {pol_b}", value=f"{total_b:,}")
            st.caption(f"Peak: {peak_b.strftime('%d.%m.%Y')} ({int(pivot[pol_b].max()):,} mentions)")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Keywords
# ══════════════════════════════════════════════════════════════════════════════

with tab_keywords:
    st.title("🔍 Keywords Mention Monitoring")

    tm = load_topic_mentions()

    # ── Chart 1: Mentions by category over time ───────────────────────────────
    st.markdown("---")
    st.subheader("Mentions by Category Over Time")

    tm_cat = _channel_filter(tm, key="cat_channel")

    cat_daily = (
        tm_cat.groupby(["date", "category"], as_index=False)["post_count"]
        .sum()
        .sort_values("date")
    )

    fig_cat = px.line(
        cat_daily, x="date", y="post_count", color="category",
        labels={"date": "Date", "post_count": "Post count", "category": "Category"},
        template="plotly_white",
    )
    fig_cat.update_traces(line_width=2)
    fig_cat.update_layout(**_CHART_LAYOUT)
    st.plotly_chart(fig_cat, use_container_width=True)

    # ── Chart 2: Most active channels ────────────────────────────────────────
    st.markdown("---")
    st.subheader("Most Mentioned Telegram Channels")

    channel_totals = (
        tm.groupby("username", as_index=False)["post_count"]
        .sum()
        .sort_values("post_count", ascending=False)
    )

    fig_bar = px.bar(
        channel_totals, x="username", y="post_count",
        labels={"username": "Channel", "post_count": "Total posts"},
        template="plotly_white",
        color_discrete_sequence=["#2196F3"],
    )
    fig_bar.update_layout(template="plotly_white", margin=dict(t=40, b=80), height=420, xaxis_tickangle=-45)
    st.plotly_chart(fig_bar, use_container_width=True)

    # ── Chart 3: Subcategories within a selected category ────────────────────
    st.markdown("---")
    st.subheader("Subcategory Breakdown by Category")

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
            subcat_daily, x="date", y="post_count", color="subcategory",
            labels={"date": "Date", "post_count": "Post count", "subcategory": "Subcategory"},
            template="plotly_white",
        )
        fig_subcat.update_traces(line_width=2)
        fig_subcat.update_layout(**_CHART_LAYOUT)
        st.plotly_chart(fig_subcat, use_container_width=True)

    # ── Chart 4: Top-5 subcategories overall ─────────────────────────────────
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
        top5_daily, x="date", y="post_count", color="subcategory",
        labels={"date": "Date", "post_count": "Post count", "subcategory": "Subcategory"},
        template="plotly_white",
    )
    fig_top5.update_traces(line_width=2)
    fig_top5.update_layout(**_CHART_LAYOUT)
    st.plotly_chart(fig_top5, use_container_width=True)

    # ── Chart 5: Heatmap — top-10 channels × categories ──────────────────────
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

    heatmap_pivot = heatmap_df.pivot_table(
        index="username", columns="category", values="post_count", fill_value=0
    )
    heatmap_pivot = heatmap_pivot.reindex(top10_channels)

    fig_heat = go.Figure(data=go.Heatmap(
        z=heatmap_pivot.values,
        x=heatmap_pivot.columns.tolist(),
        y=heatmap_pivot.index.tolist(),
        colorscale="Blues",
        hoverongaps=False,
        hovertemplate="Channel: %{y}<br>Category: %{x}<br>Posts: %{z}<extra></extra>",
    ))
    fig_heat.update_layout(
        template="plotly_white",
        margin=dict(t=40, b=100, l=160),
        height=480,
        xaxis=dict(tickangle=-30),
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    # ── Keyword Definitions ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("What Each Category Captures")
    st.caption(
        "Each category monitors specific narratives. "
        "Expand a category below to see which subcategories and sample phrases are tracked."
    )

    try:
        taxonomy = load_keyword_taxonomy()

        # Group subcategories by category
        categories_ordered: dict[str, list[dict]] = {}
        for entry in taxonomy:
            cat = entry["category"]
            if cat not in categories_ordered:
                categories_ordered[cat] = []
            categories_ordered[cat].append(entry)

        for cat_name, subcats in categories_ordered.items():
            cat_id = subcats[0]["category_id"]
            with st.expander(f"**Category {cat_id}: {cat_name}**"):
                for sc in subcats:
                    st.markdown(f"**{sc['subcategory_id']}  {sc['subcategory']}**")
                    sample = sc["phrases"][:10]
                    phrase_str = ", ".join(f"`{p}`" for p in sample)
                    if len(sc["phrases"]) > 10:
                        phrase_str += f" *(+{len(sc['phrases']) - 10} more)*"
                    st.markdown(phrase_str)
                    st.markdown("")
    except Exception as e:
        st.warning(f"Could not load keyword definitions: {e}")


# ══════════════════════════════════════════════════════════════════════════════
# TAB — Audience Insights
# ══════════════════════════════════════════════════════════════════════════════

with tab_audience:
    st.title("📡 Audience Insights")

    ai_raw = load_audience_insights()
    ct_ai = load_channel_types()

    if ai_raw is None:
        st.warning(
            "Audience Insights data not found.  \n"
            "Run `python process_audience_insights.py` first, then commit "
            "`data/audience_insights.csv` to git."
        )
        st.stop()

    # ── Join with channel types ───────────────────────────────────────────────
    if not ct_ai.empty:
        ai = ai_raw.merge(ct_ai[["username", "channel_type"]], on="username", how="left")
        ai["channel_type"] = ai["channel_type"].fillna("other")
    else:
        ai = ai_raw.copy()
        ai["channel_type"] = "other"

    # ── Channel type filter ───────────────────────────────────────────────────
    type_options = ["All types"] + sorted(ai["channel_type"].unique().tolist())
    selected_ai_type = st.selectbox("Filter by channel type", type_options, key="ai_channel_type")
    if selected_ai_type != "All types":
        ai = ai[ai["channel_type"] == selected_ai_type]

    if ai.empty:
        st.warning("No data for this channel type.")
        st.stop()

    from process_audience_insights import MONTHS as _AI_MONTHS
    st.caption(
        f"Based on the last {_AI_MONTHS} months of data · "
        f"{len(ai):,} channels in this view"
    )

    # ── Global baseline ───────────────────────────────────────────────────────
    global_posts = ai["total_posts"].sum()
    global_views = ai["total_views"].sum()
    global_avg_views = global_views / global_posts if global_posts > 0 else 1

    # ── Theme Amplification ───────────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Theme Amplification")
    st.markdown(
        "Which themes get more views than average?  \n"
        "**Amplification > 1.0** means higher-than-average reach for that theme's posts."
    )

    amp_rows = []
    for col in CATEGORY_COLS:
        posts_col = f"{col}_posts"
        views_col = f"{col}_views"
        if posts_col not in ai.columns:
            continue
        cat_posts = int(ai[posts_col].sum())
        cat_views = int(ai[views_col].sum())
        if cat_posts == 0:
            continue
        cat_avg = cat_views / cat_posts
        amp_rows.append({
            "Category": CATEGORY_LABELS[col],
            "Posts with theme": f"{cat_posts:,}",
            "Avg views": f"{cat_avg:,.0f}",
            "Amplification Factor": round(cat_avg / global_avg_views, 2),
        })

    amp_df = pd.DataFrame(amp_rows).sort_values("Amplification Factor", ascending=False).reset_index(drop=True)

    fig_amp = px.bar(
        amp_df, x="Category", y="Amplification Factor",
        color="Amplification Factor",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=1.0,
        template="plotly_white",
    )
    fig_amp.add_hline(y=1.0, line_dash="dash", annotation_text="Corpus average")
    fig_amp.update_layout(margin=dict(t=40, b=60), height=400, showlegend=False)
    st.plotly_chart(fig_amp, use_container_width=True)
    st.dataframe(amp_df, use_container_width=True, hide_index=True)

    # ── Audience Resonance Lift ───────────────────────────────────────────────
    st.markdown("---")
    st.subheader("Audience Resonance Lift")

    ai_with_avg = ai.copy()
    ai_with_avg["avg_views_per_post"] = (
        ai_with_avg["total_views"] / ai_with_avg["total_posts"].replace(0, 1)
    )
    median_avg = float(ai_with_avg["avg_views_per_post"].median())

    st.markdown(
        f"Splits channels at the median average views/post (**{int(median_avg):,}**).  \n"
        "Themes that appear more in high-engagement channels have **Lift > 1.0** — "
        "they resonate with audiences that already attract large readership."
    )

    high_eng = ai_with_avg[ai_with_avg["avg_views_per_post"] >= median_avg]
    low_eng = ai_with_avg[ai_with_avg["avg_views_per_post"] < median_avg]

    res_rows = []
    for col in CATEGORY_COLS:
        posts_col = f"{col}_posts"
        if posts_col not in ai.columns:
            continue

        high_total = int(high_eng["total_posts"].sum())
        low_total = int(low_eng["total_posts"].sum())
        high_cat = int(high_eng[posts_col].sum())
        low_cat = int(low_eng[posts_col].sum())

        high_pct = (high_cat / high_total * 100) if high_total > 0 else 0.0
        low_pct = (low_cat / low_total * 100) if low_total > 0 else 0.0
        lift = round(high_pct / low_pct, 2) if low_pct > 0 else 0.0

        res_rows.append({
            "Theme": CATEGORY_LABELS[col],
            "% in high-engagement channels": round(high_pct, 2),
            "% in low-engagement channels": round(low_pct, 2),
            "Resonance Lift": lift,
        })

    res_df = pd.DataFrame(res_rows).sort_values("Resonance Lift", ascending=False).reset_index(drop=True)

    fig_res = px.bar(
        res_df, x="Theme", y="Resonance Lift",
        color="Resonance Lift",
        color_continuous_scale="RdYlGn",
        color_continuous_midpoint=1.0,
        template="plotly_white",
    )
    fig_res.add_hline(y=1.0, line_dash="dash", annotation_text="Baseline")
    fig_res.update_layout(margin=dict(t=40, b=60), height=400, showlegend=False)
    st.plotly_chart(fig_res, use_container_width=True)
    st.dataframe(res_df, use_container_width=True, hide_index=True)
