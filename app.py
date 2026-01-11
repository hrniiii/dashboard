
# app.py
import os
import math
import traceback
import numpy as np
import pandas as pd
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, callback
import dash_bootstrap_components as dbc

# Model & scaling
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# =========================
# 1) BACA DATA & PERSIAPAN
# =========================
DATA_PATH = "ga_website_clean.csv"
df = pd.read_csv(DATA_PATH)  # pastikan file CSV ini ada di folder yang sama

# Parse kolom tanggal & konversi numerik
if "date" in df.columns:
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
for col in [
    "sessions", "pageviews", "bounces", "timeOnSite",
    "transactions", "revenue", "bounce_rate", "is_organic", "is_mobile"
]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Nilai unik untuk dropdown
sources = sorted(df["source"].dropna().unique().tolist()) if "source" in df.columns else []
mediums = sorted(df["medium"].dropna().unique().tolist()) if "medium" in df.columns else []
devices = sorted(df["device"].dropna().unique().tolist()) if "device" in df.columns else []
countries = sorted(df["country"].dropna().unique().tolist()) if "country" in df.columns else []

# Daftar variabel numerik untuk Regression selector
numeric_cols = [c for c in [
    "sessions","pageviews","timeOnSite","bounces",
    "transactions","revenue","bounce_rate"
] if c in df.columns]
if not numeric_cols:
    numeric_cols = ["sessions","pageviews"]  # fallback nama umum

# ==============================
# Tambahan: PAIR REGRESI (mengikuti app (1))
# ==============================
VALID_REGRESSION_PAIRS = {
    "sessions": ["pageviews", "transactions", "revenue"],
    "pageviews": ["transactions", "revenue"],
    "bounce_rate": ["transactions", "revenue"],
    "avg_session_duration": ["transactions", "revenue"],
    "pages_per_session": ["transactions", "revenue"],
    "transactions": ["revenue"]
}

# ==============================
# 2) INISIALISASI APLIKASI DASH
# ==============================
# suppress_callback_exceptions: boleh refer komponen walau tidak terlihat,
# tapi komponen harus ADA di layout. Kita buat kontrol Regression statis di layout.
app = Dash(__name__, external_stylesheets=[dbc.themes.LUX], suppress_callback_exceptions=True)
server = app.server

# ====================
# 3) KOMPONEN FILTER
# ====================
filters = dbc.Card(
    dbc.CardBody([
        html.H5("FILTER", className="card-title"),
        dcc.DatePickerRange(
            id="date_range",
            start_date=df["date"].min() if "date" in df.columns else None,
            end_date=df["date"].max() if "date" in df.columns else None,
            display_format="DD MMM YYYY",
            clearable=True
        ),
        html.Br(), html.Br(),
        dcc.Dropdown(id="source_dd",
                     options=[{"label": s, "value": s} for s in sources],
                     multi=True, placeholder="Pilih source"),
        html.Br(),
        dcc.Dropdown(id="medium_dd",
                     options=[{"label": m, "value": m} for m in mediums],
                     multi=True, placeholder="Pilih medium"),
        html.Br(),
        dcc.Dropdown(id="device_dd",
                     options=[{"label": d, "value": d} for d in devices],
                     multi=True, placeholder="Pilih device"),
        html.Br(),
        dcc.Dropdown(id="country_dd",
                     options=[{"label": c, "value": c} for c in countries],
                     multi=True, placeholder="Pilih negara"),
        html.Hr(),
        dbc.Checklist(
            id="flag_filters",
            options=[{"label": "Organic only", "value": "organic"},
                     {"label": "Mobile only", "value": "mobile"}],
            value=[], switch=True
        ),
    ]), className="mb-3 shadow-sm"
)

# ==============
# 4) KPI CARDS
# ==============
def kpi_card(title, id_value, color="primary"):
    return dbc.Card(
        dbc.CardBody([
            html.H6(title, className="card-title"),
            html.H3(id=id_value, className=f"text-{color}")
        ]),
        className="shadow-sm"
    )

kpi_row = dbc.Row([
    dbc.Col(kpi_card("Sessions", "kpi_sessions", "primary"), md=2),
    dbc.Col(kpi_card("Pageviews", "kpi_pageviews", "success"), md=2),
    dbc.Col(kpi_card("Bounce Rate", "kpi_bounce", "danger"), md=3),
    dbc.Col(kpi_card("Avg. Duration","kpi_duration", "info"), md=3),
    dbc.Col(kpi_card("Transactions", "kpi_tx", "warning"), md=2),
    dbc.Col(kpi_card("Revenue", "kpi_rev", "secondary"), md=3),
], className="g-3")

# ============
# 5) TABS UI
# ============
tabs = dbc.Tabs(
    [
        dbc.Tab(label="Overview", tab_id="tab-overview"),
        dbc.Tab(label="Channels", tab_id="tab-channels"),
        dbc.Tab(label="Devices", tab_id="tab-devices"),
        dbc.Tab(label="Geography", tab_id="tab-geo"),
        dbc.Tab(label="SEO", tab_id="tab-seo"),  # NEW
        dbc.Tab(label="Regression", tab_id="tab-regression"),
        dbc.Tab(label="Segmentation", tab_id="tab-segmentation"),
    ],
    id="tabs", active_tab="tab-overview"
)

# ============================
# 5.1) REGRESSION CONTROLS (STATIS DI LAYOUT)
# ============================
reg_controls = dbc.Card(
    dbc.CardBody([
        html.Div([
            dbc.Row([
                dbc.Col([
                    html.Label("X Variable"),
                    dcc.Dropdown(
                        id="reg_x",
                        options=[{"label": c, "value": c} for c in numeric_cols],
                        value=("sessions" if "sessions" in numeric_cols else (numeric_cols[0] if numeric_cols else None)),
                        clearable=False
                    )
                ], md=4),
                dbc.Col([
                    html.Label("Y Variable"),
                    # Diselaraskan: opsi Y akan diisi via callback update_reg_y
                    dcc.Dropdown(id="reg_y", clearable=False)
                ], md=4),
                dbc.Col([
                    html.Label("Aggregation"),
                    dcc.Dropdown(
                        id="reg_agg",
                        options=[
                            {"label":"Raw (row-level)", "value":"raw"},
                            {"label":"Per Day", "value":"per_day"},
                            {"label":"Per Channel", "value":"per_channel"},
                            {"label":"Per Country", "value":"per_country"},
                        ],
                        value="per_day",
                        clearable=False
                    )
                ], md=4),
            ])
        ])
    ]),
    id="reg_controls",
    className="mb-3",
    style={"display": "none"}  # default disembunyikan; akan ditampilkan saat tab-regression
)

# ===============
# 6) APP LAYOUT
# ===============
app.layout = dbc.Container([
    dbc.Row([dbc.Col(html.H2("Website Performance & Growth Insights"), md=12)]),
    dbc.Row([
        dbc.Col(filters, md=3),
        dbc.Col([kpi_row, tabs, reg_controls, html.Div(id="graphs_container")], md=9)
    ], className="mt-2")
], fluid=True)

# ==========================
# 7) HELPER & FORMATTER
# ==========================
def apply_filters(data, start_date, end_date, srcs, meds, devs, cnts, flags):
    dff = data.copy()
    if "date" in dff.columns:
        if start_date: dff = dff[dff["date"] >= pd.to_datetime(start_date)]
        if end_date: dff = dff[dff["date"] <= pd.to_datetime(end_date)]
    if srcs and "source" in dff.columns: dff = dff[dff["source"].isin(srcs)]
    if meds and "medium" in dff.columns: dff = dff[dff["medium"].isin(meds)]
    if devs and "device" in dff.columns: dff = dff[dff["device"].isin(devs)]
    if cnts and "country" in dff.columns: dff = dff[dff["country"].isin(cnts)]
    if "organic" in flags and "is_organic" in dff.columns: dff = dff[dff["is_organic"] == 1]
    if "mobile" in flags and "is_mobile" in dff.columns: dff = dff[dff["is_mobile"] == 1]
    return dff

def fmt_number(x):
    try: return f"{int(x):,}"
    except: return "0"

def fmt_currency(x):
    try: return f"${float(x):,.2f}"
    except: return "$0.00"

def fmt_duration(seconds):
    """Format detik → 'Xm YYs', aman terhadap NaN/None/negatif."""
    try:
        if seconds is None: return "–"
        if isinstance(seconds, float) and math.isnan(seconds): return "–"
        seconds = float(seconds)
        if seconds < 0: return "–"
    except Exception:
        return "–"
    m, s = divmod(int(seconds), 60)
    return f"{m:d}m {s:02d}s"

def make_reg_df(dff, agg_type, x_col, y_col):
    """Siapkan dataset regresi sesuai aggregation: raw/per_day/per_channel/per_country."""
    def agg_func(col):  # bounce_rate → mean; lainnya → sum
        return "mean" if col == "bounce_rate" else "sum"

    if agg_type == "raw":
        reg = dff[[x_col, y_col]].copy()
    elif agg_type == "per_day" and "date" in dff.columns:
        reg = (dff.groupby("date", as_index=False)
               .agg({x_col: agg_func(x_col), y_col: agg_func(y_col)}))
    elif agg_type == "per_channel" and {"source","medium"}.issubset(dff.columns):
        reg = (dff.groupby(["source","medium"], as_index=False)
               .agg({x_col: agg_func(x_col), y_col: agg_func(y_col)}))
    elif agg_type == "per_country" and "country" in dff.columns:
        reg = (dff.groupby("country", as_index=False)
               .agg({x_col: agg_func(x_col), y_col: agg_func(y_col)}))
    else:
        reg = dff[[x_col, y_col]].copy()  # fallback raw

    # Bersihkan NaN/inf
    for c in [x_col, y_col]:
        reg[c] = pd.to_numeric(reg[c], errors="coerce")
    reg = reg.replace([np.inf, -np.inf], np.nan).dropna(subset=[x_col, y_col])
    return reg

# =========================
# Tambahan: CALLBACK isi opsi Y sesuai X (mengikuti app (1))
# =========================
@app.callback(
    Output("reg_y", "options"),
    Output("reg_y", "value"),
    Input("reg_x", "value")
)
def update_reg_y(x):
    if x not in VALID_REGRESSION_PAIRS:
        return [], None
    ys = VALID_REGRESSION_PAIRS[x]
    return [{"label": y, "value": y} for y in ys], ys[0]

# =========================
# 8) CALLBACK UTAMA DASH
# =========================
@callback(
    [
        Output("kpi_sessions", "children"),
        Output("kpi_pageviews", "children"),
        Output("kpi_bounce", "children"),
        Output("kpi_duration", "children"),
        Output("kpi_tx", "children"),
        Output("kpi_rev", "children"),
        Output("graphs_container", "children"),
        Output("reg_controls", "style"),  # NEW: tampil/sembunyi kontrol regression
    ],
    [
        Input("date_range", "start_date"),
        Input("date_range", "end_date"),
        Input("source_dd", "value"),
        Input("medium_dd", "value"),
        Input("device_dd", "value"),
        Input("country_dd", "value"),
        Input("flag_filters","value"),
        Input("tabs", "active_tab"),
        Input("reg_x", "value"),
        Input("reg_y", "value"),
        Input("reg_agg", "value"),
    ]
)
def update_dashboard(start_date, end_date, srcs, meds, devs, cnts, flags,
                     active_tab, reg_x, reg_y, reg_agg):

    dff = apply_filters(df, start_date, end_date, srcs, meds, devs, cnts, flags)

    # === KPI (NaN-safe) ===
    sessions = dff["sessions"].sum() if "sessions" in dff.columns else 0
    pageviews = dff["pageviews"].sum() if "pageviews" in dff.columns else 0
    if {"bounces","sessions"}.issubset(dff.columns) and sessions > 0:
        brate = (float(dff["bounces"].sum()) / float(sessions)) * 100.0
    elif "bounce_rate" in dff.columns and len(dff) > 0:
        mean_br = dff["bounce_rate"].mean()
        brate = (mean_br * 100.0) if (isinstance(mean_br, (int,float)) and mean_br <= 1) else float(mean_br)
    else:
        brate = 0.0
    avg_dur = dff["timeOnSite"].mean() if "timeOnSite" in dff.columns and len(dff) > 0 else None
    tx = dff["transactions"].sum() if "transactions" in dff.columns else 0
    rev = dff["revenue"].sum() if "revenue" in dff.columns else 0.0

    kpi_sessions = fmt_number(sessions)
    kpi_pageviews = fmt_number(pageviews)
    kpi_bounce = f"{brate:.1f}%"
    kpi_duration = fmt_duration(avg_dur)
    kpi_tx = fmt_number(tx)
    kpi_rev = fmt_currency(rev)

    graphs = []
    reg_controls_style = {"display": "none"}  # default disembunyikan

    # Overview: tren harian Sessions & Pageviews
    if active_tab == "tab-overview":
        if "date" in dff.columns and {"sessions","pageviews"}.issubset(dff.columns):
            agg = (dff.groupby("date", as_index=False)
                   .agg({"sessions":"sum","pageviews":"sum"}))
            fig_trend = px.line(
                agg, x="date", y=["sessions","pageviews"],
                title="Tren Harian: Sessions & Pageviews",
                template="plotly_white"
            )
        else:
            fig_trend = px.scatter(title="Kolom tanggal/sessions/pageviews tidak lengkap", template="plotly_white")
        graphs.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_trend), md=12)]))

    # Channels: Source–Medium by Sessions
    elif active_tab == "tab-channels":
        if {"source","medium","sessions"}.issubset(dff.columns):
            chan = (dff.groupby(["source","medium"], as_index=False)
                    .agg({"sessions":"sum"})
                    .sort_values("sessions", ascending=False)
                    .head(20))
            fig_chan = px.bar(
                chan, x="source", y="sessions", color="medium",
                title="Top Channels (Sessions)",
                template="plotly_white", barmode="group"
            )
        else:
            fig_chan = px.scatter(title="Kolom source/medium/sessions tidak lengkap", template="plotly_white")
        graphs.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_chan), md=12)]))

    # Devices: pie sessions
    elif active_tab == "tab-devices":
        if {"device","sessions"}.issubset(dff.columns):
            dev = dff.groupby("device", as_index=False).agg({"sessions":"sum"})
            fig_dev = px.pie(
                dev, names="device", values="sessions",
                title="Distribusi Sessions per Device",
                template="plotly_white"
            )
        else:
            fig_dev = px.scatter(title="Kolom device/sessions tidak lengkap", template="plotly_white")
        graphs.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_dev), md=12)]))

    # Geography: choropleth negara (sessions)
    elif active_tab == "tab-geo":
        if {"country","sessions"}.issubset(dff.columns):
            geo = dff.groupby("country", as_index=False).agg({"sessions":"sum"})
            fig_geo = px.choropleth(
                geo, locations="country", locationmode="country names",
                color="sessions", title="Sessions per Country",
                color_continuous_scale="Blues", template="plotly_white"
            )
        else:
            fig_geo = px.scatter(title="Kolom country/sessions tidak lengkap", template="plotly_white")
        graphs.append(dbc.Row([dbc.Col(dcc.Graph(figure=fig_geo), md=12)]))

    # NEW: SEO — Compare Organic vs CPC vs Referral
    elif active_tab == "tab-seo":
        if "medium" in dff.columns:
            focus = ["organic", "cpc", "referral"]
            seo_df = dff[dff["medium"].isin(focus)].copy()
            if len(seo_df) == 0:
                graphs = [html.Div("Tidak ada data untuk medium Organic/CPC/Referral pada filter ini.", className="text-warning")]
            else:
                agg = seo_df.groupby("medium", as_index=False).agg({
                    "sessions":"sum", "pageviews":"sum",
                    "bounces":"sum" if "bounces" in seo_df.columns else "sum",
                    "transactions":"sum" if "transactions" in seo_df.columns else "sum",
                    "revenue":"sum" if "revenue" in seo_df.columns else "sum",
                    "bounce_rate":"mean" if "bounce_rate" in seo_df.columns else "mean"
                })
                agg["pages_per_session"] = agg["pageviews"] / agg["sessions"]
                agg["conversion_rate"] = (agg["transactions"] / agg["sessions"]) * 100.0 if "transactions" in agg.columns else 0.0
                if "bounces" in agg.columns:
                    agg["bounce_rate_pct"] = (agg["bounces"] / agg["sessions"]) * 100.0
                else:
                    agg["bounce_rate_pct"] = agg["bounce_rate"].apply(lambda x: x*100.0 if x<=1 else x)
                agg["rps"] = agg["revenue"] / agg["sessions"] if "revenue" in agg.columns else 0.0
                agg["aov"] = agg["revenue"] / agg["transactions"] if "transactions" in agg.columns else 0.0

                # Volume
                vol_melt = agg.melt(id_vars="medium", value_vars=["sessions","pageviews"],
                                    var_name="metric", value_name="value")
                fig_vol = px.bar(vol_melt, x="medium", y="value", color="metric",
                                 title="Volume: Sessions & Pageviews (Organic vs CPC vs Referral)",
                                 template="plotly_white", barmode="group")
                # Rates
                rate_melt = agg.melt(id_vars="medium",
                                     value_vars=["bounce_rate_pct","conversion_rate"],
                                     var_name="metric", value_name="value")
                fig_rate = px.bar(rate_melt, x="medium", y="value", color="metric",
                                  title="Rates: Bounce vs Conversion",
                                  template="plotly_white", barmode="group")
                fig_rate.update_yaxes(title="Percent (%)")
                # Revenue
                if "revenue" in agg.columns:
                    fig_rev = px.bar(agg, x="medium", y="revenue",
                                     title="Revenue per Medium",
                                     template="plotly_white")
                else:
                    fig_rev = px.scatter(title="Kolom revenue tidak tersedia.", template="plotly_white")

                graphs = [
                    dbc.Row([dbc.Col(dcc.Graph(figure=fig_vol), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(figure=fig_rate), md=12)]),
                    dbc.Row([dbc.Col(dcc.Graph(figure=fig_rev), md=12)]),
                ]
        else:
            graphs = [html.Div("Kolom 'medium' tidak tersedia.", className="text-danger")]

    # Regression: diselaraskan dengan app (1) — gunakan VALID_REGRESSION_PAIRS
    elif active_tab == "tab-regression":
        reg_controls_style = {"display": "block"}  # tampilkan kontrol saat tab-regression

        # ======= DEFAULTS SESUAI PAIR =======
        if reg_x is None or reg_x not in VALID_REGRESSION_PAIRS:
            reg_x = list(VALID_REGRESSION_PAIRS.keys())[0]
        valid_y = VALID_REGRESSION_PAIRS.get(reg_x, [])
        if reg_y not in valid_y:
            reg_y = valid_y[0] if valid_y else None

        # Guard ketika Y belum siap
        if reg_x is None or reg_y is None:
            return (kpi_sessions, kpi_pageviews, kpi_bounce, kpi_duration, kpi_tx, kpi_rev,
                    html.Div(), reg_controls_style)

        try:
            if reg_agg is None: reg_agg = "per_day"
            reg = make_reg_df(dff, reg_agg, reg_x, reg_y)

            if len(reg) >= 5 and reg[reg_x].nunique() > 1:
                X = reg[[reg_x]].values
                y = reg[reg_y].values

                lr = LinearRegression().fit(X, y)
                x_line = np.linspace(float(X.min()), float(X.max()), 100).reshape(-1, 1)
                y_line = lr.predict(x_line)

                fig_reg = px.scatter(
                    reg, x=reg_x, y=reg_y,
                    labels={reg_x: reg_x, reg_y: reg_y},
                    title=f"{reg_y} ~ {reg_x} ({reg_agg})",
                    template="plotly_white"
                )
                fig_reg.add_traces(px.line(x=x_line.ravel(), y=y_line).data)

                graphs = [dbc.Row([dbc.Col(dcc.Graph(figure=fig_reg), md=12)])]
            else:
                graphs = [html.Div("Data untuk regresi belum memadai.", className="text-warning")]

        except Exception:
            graphs = [html.Pre(traceback.format_exc(), style={"whiteSpace":"pre-wrap"})]

    # Segmentation: K-Means (k=3) pada fitur numerik
    elif active_tab == "tab-segmentation":
        num_features = [c for c in ["sessions","pageviews","bounce_rate","transactions","revenue"] if c in dff.columns]
        if len(num_features) >= 2 and len(dff) >= 10:
            seg = dff[num_features].copy().dropna()
            if len(seg) < 5:
                graphs = [html.Div("Terlalu sedikit data setelah pembersihan NaN.", className="text-warning")]
            else:
                scaler = StandardScaler()
                Xs = scaler.fit_transform(seg.values)
                kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
                labels = kmeans.fit_predict(Xs)
                seg = seg.assign(cluster=labels)
                if {"sessions","pageviews"}.issubset(seg.columns):
                    hover_cols = [c for c in ["source","medium","device","country"] if c in dff.columns]
                    seg_full = dff.loc[seg.index, hover_cols].copy()
                    plot_df = pd.concat([seg[["sessions","pageviews","cluster"]], seg_full], axis=1)
                    fig_km = px.scatter(
                        plot_df, x="sessions", y="pageviews", color="cluster",
                        hover_data=hover_cols,
                        title="K-Means Segmentation (k=3): Sessions vs Pageviews",
                        template="plotly_white"
                    )
                else:
                    fig_km = px.scatter(title="Butuh sessions & pageviews untuk visualisasi utama.", template="plotly_white")
                graphs = [dbc.Row([dbc.Col(dcc.Graph(figure=fig_km), md=12)])]
        else:
            graphs = [html.Div("Clustering butuh ≥2 fitur numerik dan ≥10 baris data.", className="text-danger")]

    else:
        graphs = [html.Div("Silakan pilih tab untuk melihat visualisasi.", className="text-info")]

    return kpi_sessions, kpi_pageviews, kpi_bounce, kpi_duration, kpi_tx, kpi_rev, html.Div(graphs), reg_controls_style

# ============================
# 9) ENTRY POINT (DEV SERVER)
# ============================
if __name__ == "__main__":
    app.run(debug=True)
