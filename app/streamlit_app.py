import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings("ignore")



# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Factory Optimization",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: #f0f2f6;
        border-radius: 10px;
        padding: 16px;
        text-align: center;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; }
    .stTabs [data-baseweb="tab"] { border-radius: 6px 6px 0 0; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING & FEATURE ENGINEERING
# ─────────────────────────────────────────────
@st.cache(allow_output_mutation=True, suppress_st_warning=True)
def load_and_prepare():
    import os
    # Support both .xlsx and .csv, spaces or underscores in filename
    base = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.path.join(base, "../data/Nassau Candy Distributor.xlsx"),
        os.path.join(base, "../data/Nassau_Candy_Distributor.xlsx"),
        os.path.join(base, "../data/Nassau Candy Distributor.csv"),
        os.path.join(base, "../data/Nassau_Candy_Distributor.csv"),
        os.path.join(base, "data/Nassau Candy Distributor.xlsx"),
        os.path.join(base, "data/Nassau_Candy_Distributor.xlsx"),
        os.path.join(base, "data/Nassau Candy Distributor.csv"),
        os.path.join(base, "data/Nassau_Candy_Distributor.csv"),
    ]
    df = None
    for p in candidates:
        if os.path.exists(p):
            df = pd.read_excel(p) if p.endswith(".xlsx") else pd.read_csv(p)
            break
    if df is None:
        return None  # handled outside the cached function

    # ── Engineer realistic Lead_Time from Ship Mode + Region + Division ──
    np.random.seed(42)
    ship_mode_days  = {'Same Day': 1, 'First Class': 2, 'Second Class': 4, 'Standard Class': 7}
    region_penalty  = {'Interior': 1, 'Atlantic': 0, 'Gulf': 1, 'Pacific': 2}
    div_penalty     = {'Chocolate': 0, 'Sugar': 0, 'Other': 1}

    df['Lead_Time'] = (
        df['Ship Mode'].map(ship_mode_days) +
        df['Region'].map(region_penalty) +
        df['Division'].map(div_penalty) +
        np.random.randint(0, 3, size=len(df))
    )

    # ── Assign factories based on Region + Division (business-logic proxy) ──
    factory_map = {
        ('Chocolate', 'Atlantic') : 'Factory_East',
        ('Chocolate', 'Interior') : 'Factory_Central',
        ('Chocolate', 'Gulf')     : 'Factory_South',
        ('Chocolate', 'Pacific')  : 'Factory_West',
        ('Sugar',     'Atlantic') : 'Factory_East',
        ('Sugar',     'Interior') : 'Factory_Central',
        ('Sugar',     'Gulf')     : 'Factory_South',
        ('Sugar',     'Pacific')  : 'Factory_West',
        ('Other',     'Atlantic') : 'Factory_East',
        ('Other',     'Interior') : 'Factory_Central',
        ('Other',     'Gulf')     : 'Factory_South',
        ('Other',     'Pacific')  : 'Factory_West',
    }
    df['Current_Factory'] = df.apply(
        lambda r: factory_map.get((r['Division'], r['Region']), 'Factory_Central'), axis=1
    )

    # ── Profit Margin ──
    df['Profit_Margin'] = (df['Gross Profit'] / df['Sales']).round(4)

    return df

df = load_and_prepare()
if df is None:
    st.error("❌ Could not find the data file. Make sure **Nassau Candy Distributor.xlsx** (or .csv) is inside the `data/` folder next to `app/`.")
    st.stop()
FACTORIES = ['Factory_East', 'Factory_West', 'Factory_Central', 'Factory_South']

# ─────────────────────────────────────────────
# MODEL TRAINING
# ─────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def train_models(df):
    feature_cols = ['Division', 'Region', 'Ship Mode', 'Current_Factory',
                    'Sales', 'Units', 'Cost', 'Gross Profit', 'Profit_Margin']
    df_enc = pd.get_dummies(df[feature_cols + ['Lead_Time']], columns=['Division', 'Region', 'Ship Mode', 'Current_Factory'])
    X = df_enc.drop('Lead_Time', axis=1)
    y = df_enc['Lead_Time']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        'Linear Regression'   : LinearRegression(),
        'Random Forest'       : RandomForestRegressor(n_estimators=100, random_state=42),
        'Gradient Boosting'   : GradientBoostingRegressor(n_estimators=100, random_state=42),
    }

    results = {}
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        preds = m.predict(X_test)
        results[name] = {
            'RMSE' : round(np.sqrt(mean_squared_error(y_test, preds)), 3),
            'MAE'  : round(mean_absolute_error(y_test, preds), 3),
            'R2'   : round(r2_score(y_test, preds), 3),
        }
        trained[name] = m

    best_name = min(results, key=lambda k: results[k]['RMSE'])
    return trained, results, best_name, X.columns.tolist()

trained_models, eval_results, best_model_name, feature_cols = train_models(df)
best_model = trained_models[best_model_name]

# ─────────────────────────────────────────────
# CLUSTERING
# ─────────────────────────────────────────────
@st.cache(allow_output_mutation=True)
def cluster_routes(df):
    agg = df.groupby(['Region', 'Ship Mode', 'Division']).agg(
        Avg_Lead_Time=('Lead_Time', 'mean'),
        Avg_Profit_Margin=('Profit_Margin', 'mean'),
        Volume=('Units', 'sum'),
        Avg_Sales=('Sales', 'mean'),
    ).reset_index()

    scaler = StandardScaler()
    X_cl = scaler.fit_transform(agg[['Avg_Lead_Time', 'Avg_Profit_Margin', 'Volume']])
    km = KMeans(n_clusters=4, random_state=42, n_init=10)
    agg['Cluster'] = km.fit_predict(X_cl)

    cluster_labels = {
        agg.groupby('Cluster')['Avg_Lead_Time'].mean().idxmax()    : '🔴 Slow Routes',
        agg.groupby('Cluster')['Avg_Lead_Time'].mean().idxmin()    : '🟢 Fast Routes',
        agg.groupby('Cluster')['Avg_Profit_Margin'].mean().idxmax(): '🟡 High-Margin',
        agg.groupby('Cluster')['Avg_Profit_Margin'].mean().idxmin(): '🟠 Low-Margin',
    }
    agg['Cluster_Label'] = agg['Cluster'].map(
        lambda c: cluster_labels.get(c, f'Cluster {c}')
    )
    return agg

route_clusters = cluster_routes(df)

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60?text=Nassau+Candy", width=200)
    st.title("🍬 Nassau Candy")
    st.caption("Factory Optimization System")
    st.markdown("---")

    st.subheader("🔧 Scenario Configuration")
    sel_division  = st.selectbox("Product Division", sorted(df['Division'].unique()))
    sel_region    = st.selectbox("Destination Region", sorted(df['Region'].unique()))
    sel_ship_mode = st.selectbox("Ship Mode", sorted(df['Ship Mode'].unique()))
    opt_priority  = st.slider("Optimization Priority", 0, 100, 50,
                               help="0 = Maximize Speed | 100 = Maximize Profit")
    top_n = st.slider("Top N Recommendations", 1, 4, 3)

    st.markdown("---")
    st.caption(f"📊 Dataset: {len(df):,} orders | Best model: **{best_model_name}**")

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview",
    "🏭 Factory Optimizer",
    "🔄 What-If Simulator",
    "🎯 Recommendations",
    "⚠️ Risk & Impact",
])

# ═══════════════════════════════════════════════════════
# TAB 1 – OVERVIEW & MODEL EVALUATION
# ═══════════════════════════════════════════════════════
with tab1:
    st.header("📊 Dashboard Overview")

    # ── Section 1: FULL DATASET summary (always static — correct behavior) ──
    st.markdown("#### 🌐 Global Summary (entire dataset)")
    st.caption("These numbers always reflect all 10,194 orders — they don't change with the sidebar. "
               "Use the other tabs to see results for your selected scenario.")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Orders",      f"{len(df):,}")
    c2.metric("Avg Lead Time",     f"{df['Lead_Time'].mean():.1f} days")
    c3.metric("Avg Profit Margin", f"{df['Profit_Margin'].mean()*100:.1f}%")
    c4.metric("Best Model (R²)",   f"{eval_results[best_model_name]['R2']}")

    st.markdown("---")

    # ── Section 2: FILTERED summary — THESE change with the sidebar ──
    df_filtered = df[
        (df['Division']  == sel_division) &
        (df['Region']    == sel_region) &
        (df['Ship Mode'] == sel_ship_mode)
    ]

    st.markdown(f"#### 🔍 Filtered View — Division: `{sel_division}` | Region: `{sel_region}` | Ship Mode: `{sel_ship_mode}`")

    if len(df_filtered) == 0:
        st.warning("No orders found for this combination. Try different sidebar filters.")
    else:
        f1, f2, f3, f4 = st.columns(4)
        f1.metric("Matching Orders",
                  f"{len(df_filtered):,}",
                  delta=f"{len(df_filtered)/len(df)*100:.1f}% of total")
        f2.metric("Avg Lead Time",
                  f"{df_filtered['Lead_Time'].mean():.1f} days",
                  delta=f"{df_filtered['Lead_Time'].mean() - df['Lead_Time'].mean():.1f} vs avg",
                  delta_color="inverse")
        f3.metric("Avg Profit Margin",
                  f"{df_filtered['Profit_Margin'].mean()*100:.1f}%",
                  delta=f"{(df_filtered['Profit_Margin'].mean() - df['Profit_Margin'].mean())*100:.1f}% vs avg")
        f4.metric("Total Sales",
                  f"${df_filtered['Sales'].sum():,.0f}")

        col_a, col_b = st.columns(2)
        with col_a:
            st.subheader("Lead Time by Ship Mode (Filtered Division & Region)")
            df_div_reg = df[(df['Division'] == sel_division) & (df['Region'] == sel_region)]
            fig = px.box(df_div_reg, x='Ship Mode', y='Lead_Time', color='Ship Mode',
                         title=f"Lead Time — {sel_division} / {sel_region}")
            st.plotly_chart(fig)

        with col_b:
            st.subheader("Profit Margin by Factory (Filtered Division)")
            df_div = df[df['Division'] == sel_division]
            fig2 = px.violin(df_div, x='Current_Factory', y='Profit_Margin',
                             color='Current_Factory',
                             title=f"Profit Margin by Factory — {sel_division}", box=True)
            st.plotly_chart(fig2)

    st.markdown("---")

    # ── Section 3: Global charts (always full dataset — for context) ──
    st.markdown("#### 📊 Global Distribution Charts")

    col_a2, col_b2 = st.columns(2)
    with col_a2:
        fig_g1 = px.box(df, x='Ship Mode', y='Lead_Time', color='Ship Mode',
                        title="Lead Time Distribution per Ship Mode (All Data)")
        st.plotly_chart(fig_g1)
    with col_b2:
        fig_g2 = px.violin(df, x='Region', y='Profit_Margin', color='Region',
                           title="Profit Margin Distribution per Region (All Data)", box=True)
        st.plotly_chart(fig_g2)

    st.subheader("🤖 Model Evaluation Comparison")
    eval_df = pd.DataFrame(eval_results).T.reset_index().rename(columns={'index': 'Model'})
    eval_df['Best'] = eval_df['Model'] == best_model_name

    fig3 = go.Figure()
    for metric in ['RMSE', 'MAE', 'R2']:
        fig3.add_trace(go.Bar(
            name=metric, x=eval_df['Model'], y=eval_df[metric],
            text=eval_df[metric], textposition='outside'
        ))
    fig3.update_layout(barmode='group', title="RMSE / MAE / R² across Models",
                       legend_title="Metric")
    st.plotly_chart(fig3)

    st.info(f"✅ **Best performing model: {best_model_name}** — lowest RMSE, used for all predictions below.")

    st.subheader("📍 Route Clustering Map")
    fig4 = px.scatter(route_clusters,
                      x='Avg_Lead_Time', y='Avg_Profit_Margin',
                      color='Cluster_Label', size='Volume',
                      hover_data=['Region', 'Ship Mode', 'Division'],
                      title="Route Clusters: Speed vs. Profitability")
    st.plotly_chart(fig4)


# ═══════════════════════════════════════════════════════
# TAB 2 – FACTORY OPTIMIZER
# ═══════════════════════════════════════════════════════
with tab2:
    st.header("🏭 Factory Optimizer")
    st.markdown(f"**Selected:** Division=`{sel_division}` | Region=`{sel_region}` | Ship Mode=`{sel_ship_mode}`")

    def predict_lead_time(division, region, ship_mode, factory):
        sample = {
            'Sales'        : df['Sales'].mean(),
            'Units'        : df['Units'].mean(),
            'Cost'         : df['Cost'].mean(),
            'Gross Profit' : df['Gross Profit'].mean(),
            'Profit_Margin': df['Profit_Margin'].mean(),
            'Division'     : division,
            'Region'       : region,
            'Ship Mode'    : ship_mode,
            'Current_Factory': factory,
        }
        temp_df = pd.DataFrame([sample])
        temp_enc = pd.get_dummies(temp_df, columns=['Division', 'Region', 'Ship Mode', 'Current_Factory'])
        temp_enc = temp_enc.reindex(columns=feature_cols, fill_value=0)
        return best_model.predict(temp_enc)[0]

    # ── Build predictions for all 4 factories ──
    # Each factory has a realistic operational cost profile that affects margin
    # Fast factories = higher logistics cost = lower margin, and vice versa
    factory_cost_profile = {
        'Factory_East'    : {'margin_boost': +0.08, 'label': 'Premium (fast, costly)'},
        'Factory_West'    : {'margin_boost': +0.14, 'label': 'Budget (slow, cheap)'},
        'Factory_Central' : {'margin_boost': +0.05, 'label': 'Balanced'},
        'Factory_South'   : {'margin_boost': +0.11, 'label': 'Economy (moderate)'},
    }

    sim_results = []
    base_margin = df[
        (df['Division'] == sel_division) & (df['Region'] == sel_region)
    ]['Profit_Margin'].mean()

    for fac in FACTORIES:
        lt = predict_lead_time(sel_division, sel_region, sel_ship_mode, fac)
        # Margin: fast factories cost more to operate → lower margin
        # Slow factories have lower logistics cost → higher margin
        profile = factory_cost_profile.get(fac, {'margin_boost': 0.05})
        est_margin = base_margin + profile['margin_boost']
        sim_results.append({
            'Factory'                   : fac,
            'Factory Profile'           : profile['label'],
            'Predicted Lead Time (days)': round(lt, 2),
            'Est. Profit Margin (%)'    : round(min(est_margin * 100, 99), 1),
        })

    sim_df = pd.DataFrame(sim_results)

    # ── Apply Optimization Priority to rank factories ──
    # priority=0 → pure speed, priority=100 → pure profit
    speed_w  = 1 - opt_priority / 100
    profit_w = opt_priority / 100

    lt_vals = sim_df['Predicted Lead Time (days)']
    pm_vals = sim_df['Est. Profit Margin (%)']

    lt_min, lt_max = lt_vals.min(), lt_vals.max()
    pm_min, pm_max = pm_vals.min(), pm_vals.max()

    def norm_speed(v):   # lower lead time = higher score
        return 1 - (v - lt_min) / (lt_max - lt_min + 1e-9)
    def norm_profit(v):  # higher margin = higher score
        return (v - pm_min) / (pm_max - pm_min + 1e-9)

    sim_df['Speed Score (0-1)']  = sim_df['Predicted Lead Time (days)'].apply(norm_speed).round(3)
    sim_df['Profit Score (0-1)'] = sim_df['Est. Profit Margin (%)'].apply(norm_profit).round(3)
    sim_df['Combined Score']     = (speed_w * sim_df['Speed Score (0-1)'] +
                                    profit_w * sim_df['Profit Score (0-1)']).round(3)

    sim_df = sim_df.sort_values('Combined Score', ascending=False).reset_index(drop=True)

    best_fac  = sim_df.iloc[0]
    worst_fac = sim_df.iloc[-1]
    improvement = round(
        sim_df['Predicted Lead Time (days)'].max() - best_fac['Predicted Lead Time (days)'], 2
    )

    # ── Show what mode we are in ──
    if opt_priority <= 20:
        mode_label = "⚡ Speed Priority"
    elif opt_priority >= 80:
        mode_label = "💰 Profit Priority"
    else:
        mode_label = "⚖️ Balanced"

    st.info(f"**Optimization Mode:** {mode_label} — Speed weight: {speed_w:.0%} | Profit weight: {profit_w:.0%}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🏆 Best Factory",    best_fac['Factory'])
    c2.metric("⏱️ Lead Time",       f"{best_fac['Predicted Lead Time (days)']} days")
    c3.metric("💰 Profit Margin",   f"{best_fac['Est. Profit Margin (%)']}%")
    c4.metric("📉 Time Saved",      f"{improvement} days vs worst")

    st.subheader("📋 All Factory Predictions (ranked by your priority setting)")
    display_cols = ['Factory', 'Factory Profile', 'Predicted Lead Time (days)',
                    'Est. Profit Margin (%)', 'Speed Score (0-1)', 'Profit Score (0-1)', 'Combined Score']
    st.dataframe(sim_df[display_cols])

    # ── Chart 1: Actual lead times as vertical bars ──
    st.subheader("📊 Predicted Lead Time per Factory")
    lt_sorted = sim_df.sort_values('Predicted Lead Time (days)')
    y_min = max(0, lt_sorted['Predicted Lead Time (days)'].min() - 1)
    y_max = lt_sorted['Predicted Lead Time (days)'].max() + 1

    fig5 = px.bar(lt_sorted,
                  x='Factory',
                  y='Predicted Lead Time (days)',
                  color='Factory',
                  text='Predicted Lead Time (days)',
                  title="Lower = Better | Sorted fastest → slowest",
                  color_discrete_sequence=px.colors.qualitative.Set2)
    fig5.update_traces(texttemplate='%{text:.2f} days', textposition='outside')
    fig5.update_layout(yaxis=dict(range=[0, y_max + 1]),
                       yaxis_title="Lead Time (days)",
                       showlegend=False)
    st.plotly_chart(fig5)

    # ── Chart 2: Combined score bar — shows priority effect ──
    st.subheader("🏆 Combined Score (reflects your Optimization Priority slider)")
    fig6 = px.bar(sim_df,
                  x='Factory',
                  y='Combined Score',
                  color='Factory',
                  text='Combined Score',
                  title=f"Combined Score at Priority={opt_priority} — Higher = Recommended",
                  color_discrete_sequence=px.colors.qualitative.Pastel)
    fig6.update_traces(texttemplate='%{text:.3f}', textposition='outside')
    fig6.update_layout(yaxis=dict(range=[0, 1.2]),
                       yaxis_title="Score (0 to 1)",
                       showlegend=False)
    st.plotly_chart(fig6)

    # ── Chart 3: Side-by-side speed vs profit ──
    st.subheader("📊 Speed vs Profit — Side by Side")
    col_f1, col_f2 = st.columns(2)

    with col_f1:
        fig6a = px.bar(lt_sorted,
                       x='Predicted Lead Time (days)', y='Factory',
                       orientation='h', color='Factory',
                       text='Predicted Lead Time (days)',
                       title="🏎️ Fastest → Slowest",
                       color_discrete_sequence=px.colors.qualitative.Set2)
        fig6a.update_traces(texttemplate='%{text:.2f}d', textposition='inside')
        fig6a.update_layout(showlegend=False, xaxis=dict(range=[y_min, y_max]))
        st.plotly_chart(fig6a)

    with col_f2:
        pm_sorted = sim_df.sort_values('Est. Profit Margin (%)', ascending=False)
        p_min = max(0, pm_sorted['Est. Profit Margin (%)'].min() - 1)
        p_max = pm_sorted['Est. Profit Margin (%)'].max() + 1
        fig6b = px.bar(pm_sorted,
                       x='Est. Profit Margin (%)', y='Factory',
                       orientation='h', color='Factory',
                       text='Est. Profit Margin (%)',
                       title="💰 Most → Least Profitable",
                       color_discrete_sequence=px.colors.qualitative.Pastel)
        fig6b.update_traces(texttemplate='%{text:.1f}%', textposition='inside')
        fig6b.update_layout(showlegend=False, xaxis=dict(range=[p_min, p_max]))
        st.plotly_chart(fig6b)


# ═══════════════════════════════════════════════════════
# TAB 3 – WHAT-IF SIMULATOR
# ═══════════════════════════════════════════════════════
with tab3:
    st.header("🔄 What-If Scenario Analysis")

    col_curr, col_new = st.columns(2)

    with col_curr:
        st.subheader("Current Assignment")
        curr_factory = st.selectbox("Current Factory", FACTORIES, key="curr")
        curr_lt = predict_lead_time(sel_division, sel_region, sel_ship_mode, curr_factory)
        st.metric("Current Lead Time", f"{curr_lt:.2f} days")

    with col_new:
        st.subheader("Proposed Reassignment")
        new_factory = st.selectbox("New Factory", FACTORIES, key="new",
                                   index=1 if FACTORIES[0] == curr_factory else 0)
        new_lt = predict_lead_time(sel_division, sel_region, sel_ship_mode, new_factory)
        st.metric("New Lead Time", f"{new_lt:.2f} days",
                  delta=f"{new_lt - curr_lt:.2f} days",
                  delta_color="inverse")

    delta   = curr_lt - new_lt
    pct_chg = (delta / curr_lt) * 100 if curr_lt > 0 else 0

    if delta > 0:
        st.success(f"✅ Reassigning to **{new_factory}** saves **{delta:.2f} days** ({pct_chg:.1f}% improvement)")
    elif delta < 0:
        st.warning(f"⚠️ Reassigning to **{new_factory}** adds **{-delta:.2f} days** — not recommended for speed.")
    else:
        st.info("No difference in predicted lead time.")

    # Bar comparison chart
    compare_df = pd.DataFrame([
        {'Scenario': f'Current ({curr_factory})',    'Lead Time (days)': curr_lt},
        {'Scenario': f'Proposed ({new_factory})',    'Lead Time (days)': new_lt},
    ])
    fig7 = px.bar(compare_df, x='Scenario', y='Lead Time (days)',
                  color='Scenario', text='Lead Time (days)',
                  title="Current vs Proposed Lead Time",
                  color_discrete_sequence=['#EF553B', '#00CC96'])
    fig7.update_traces(texttemplate='%{text:.2f}', textposition='outside')
    st.plotly_chart(fig7)

    # Sweep all scenarios
    st.subheader("📊 Full Scenario Sweep")
    all_combos = []
    for fac in FACTORIES:
        for sm in df['Ship Mode'].unique():
            lt = predict_lead_time(sel_division, sel_region, sm, fac)
            all_combos.append({'Factory': fac, 'Ship Mode': sm, 'Lead Time': round(lt, 2)})

    sweep_df = pd.DataFrame(all_combos)
    fig8 = px.density_heatmap(sweep_df, x='Factory', y='Ship Mode',
                               z='Lead Time', text_auto=True,
                               title="Lead Time Heatmap: Factory × Ship Mode",
                               color_continuous_scale='RdYlGn_r')
    st.plotly_chart(fig8)


# ═══════════════════════════════════════════════════════
# TAB 4 – RECOMMENDATIONS
# ═══════════════════════════════════════════════════════
with tab4:
    st.header("🎯 Factory Reassignment Recommendations")
    st.markdown("Ranked recommendations across **all Division × Region combinations**.")

    speed_weight  = 1 - opt_priority / 100
    profit_weight = opt_priority / 100

    recs = []
    for div in df['Division'].unique():
        for reg in df['Region'].unique():
            current_fac = df[(df['Division'] == div) & (df['Region'] == reg)]['Current_Factory'].mode()[0]
            avg_margin  = df[(df['Division'] == div) & (df['Region'] == reg)]['Profit_Margin'].mean()

            options = []
            for fac in FACTORIES:
                lt = predict_lead_time(div, reg, 'Standard Class', fac)
                prof = avg_margin * (1 + (10 - lt) * 0.01)
                # Normalize scores: lower lead_time = better, higher profit = better
                options.append({'factory': fac, 'lt': lt, 'profit': prof})

            # Normalize
            lt_min = min(o['lt'] for o in options)
            lt_max = max(o['lt'] for o in options)
            pr_min = min(o['profit'] for o in options)
            pr_max = max(o['profit'] for o in options)

            for o in options:
                lt_score  = 1 - (o['lt'] - lt_min) / (lt_max - lt_min + 1e-9)
                pr_score  = (o['profit'] - pr_min) / (pr_max - pr_min + 1e-9)
                composite = speed_weight * lt_score + profit_weight * pr_score
                o['composite'] = composite

            best_opt = max(options, key=lambda x: x['composite'])

            if best_opt['factory'] != current_fac:
                recs.append({
                    'Division'            : div,
                    'Region'              : reg,
                    'Current Factory'     : current_fac,
                    'Recommended Factory' : best_opt['factory'],
                    'Predicted Lead Time' : round(best_opt['lt'], 2),
                    'Est. Profit Margin %': round(best_opt['profit'] * 100, 2),
                    'Confidence Score'    : round(best_opt['composite'] * 100, 1),
                })

    recs_df = pd.DataFrame(recs).sort_values('Confidence Score', ascending=False)

    st.metric("Total Reassignment Opportunities", len(recs_df))

    top_recs = recs_df.head(top_n)
    for _, row in top_recs.iterrows():
        with st.expander(f"🏭 {row['Division']} | {row['Region']} → Reassign to **{row['Recommended Factory']}** (Score: {row['Confidence Score']}%)"):
            rc1, rc2, rc3 = st.columns(3)
            rc1.metric("Current Factory",       row['Current Factory'])
            rc2.metric("Recommended Factory",   row['Recommended Factory'])
            rc3.metric("Lead Time Saved",        f"{row['Predicted Lead Time']} days")

    st.subheader("📋 All Recommendations")
    st.dataframe(recs_df)

    fig9 = px.scatter(recs_df,
                      x='Predicted Lead Time',
                      y='Est. Profit Margin %',
                      color='Recommended Factory',
                      size='Confidence Score',
                      hover_data=['Division', 'Region'],
                      title="Recommendations: Lead Time vs Profit Impact")
    st.plotly_chart(fig9)


# ═══════════════════════════════════════════════════════
# TAB 5 – RISK & IMPACT
# ═══════════════════════════════════════════════════════
with tab5:
    st.header("⚠️ Risk & Profit Impact Analysis")

    # Slow routes
    slow_threshold  = df['Lead_Time'].quantile(0.75)
    low_margin_thresh = df['Profit_Margin'].quantile(0.25)

    slow_routes = df[df['Lead_Time'] >= slow_threshold].groupby(
        ['Division', 'Region', 'Ship Mode', 'Current_Factory']
    ).agg(
        Avg_Lead_Time=('Lead_Time', 'mean'),
        Volume=('Units', 'sum'),
        Avg_Margin=('Profit_Margin', 'mean')
    ).reset_index().sort_values('Avg_Lead_Time', ascending=False).head(15)

    low_margin_routes = df[df['Profit_Margin'] <= low_margin_thresh].groupby(
        ['Division', 'Region', 'Current_Factory']
    ).agg(
        Avg_Margin=('Profit_Margin', 'mean'),
        Volume=('Units', 'sum'),
        Avg_Lead_Time=('Lead_Time', 'mean')
    ).reset_index().sort_values('Avg_Margin').head(10)

    col_r1, col_r2 = st.columns(2)

    with col_r1:
        st.subheader("🔴 High Risk: Slow Routes")
        st.caption(f"Routes with lead time ≥ {slow_threshold:.1f} days (top 25%)")
        st.dataframe(slow_routes.style.background_gradient(
            subset=['Avg_Lead_Time'], cmap='RdYlGn_r'))

    with col_r2:
        st.subheader("🟠 Margin Risk: Low-Profit Routes")
        st.caption(f"Routes with profit margin ≤ {low_margin_thresh:.2%} (bottom 25%)")
        st.dataframe(low_margin_routes.style.background_gradient(
            subset=['Avg_Margin'], cmap='RdYlGn'))

    # Profit impact alerts
    st.subheader("💡 Profit Impact Alerts")
    combined_risk = df[(df['Lead_Time'] >= slow_threshold) & (df['Profit_Margin'] <= low_margin_thresh)]
    if len(combined_risk) > 0:
        agg_risk = combined_risk.groupby(['Division', 'Region', 'Current_Factory']).agg(
            Orders=('Sales', 'count'),
            Total_Sales=('Sales', 'sum'),
            Avg_Profit=('Profit_Margin', 'mean'),
            Avg_LT=('Lead_Time', 'mean')
        ).reset_index()
        st.warning(f"🚨 Found **{len(agg_risk)} route combinations** with BOTH high lead time AND low margin — priority reassignment targets!")
        st.dataframe(agg_risk.sort_values('Avg_LT', ascending=False))
    else:
        st.success("✅ No routes found with simultaneous slow lead time and low margin.")

    # Treemap of lead time by factory
    st.subheader("🗺️ Lead Time Exposure by Factory")
    factory_agg = df.groupby(['Current_Factory', 'Division', 'Region']).agg(
        Avg_LT=('Lead_Time', 'mean'),
        Volume=('Units', 'sum')
    ).reset_index()
    fig10 = px.treemap(factory_agg, path=['Current_Factory', 'Division', 'Region'],
                       values='Volume', color='Avg_LT',
                       color_continuous_scale='RdYlGn_r',
                       title="Lead Time by Factory → Division → Region")
    st.plotly_chart(fig10)
