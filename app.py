"""Food Preservation Safety Prediction dashboard."""
from pathlib import Path
import base64

import joblib
import pandas as pd
import streamlit as st
from analytics import get_kpis
from visualizations import (plot_box_by_target, plot_correlation_heatmap,
    plot_distribution, plot_feature_importance, plot_food_type_distribution,
    plot_food_type_preservation, plot_model_comparison, plot_microbial_scatter,
    plot_preservation_rate, plot_score_risk_categories, plot_target_distribution)

ROOT = Path(__file__).resolve().parent
BACKGROUND_PATH = ROOT / "food_safety_background.png"
FEATURES = ["food_type", "temperature_c", "humidity_percent", "storage_days", "ph_level",
            "packaging_type", "storage_condition", "microbial_count_cfu", "preservative_used",
            "expiration_days_remaining"]

st.set_page_config(page_title="Food Preservation Safety", page_icon="🍽️", layout="wide")

@st.cache_data
def background_url():
    return "data:image/png;base64," + base64.b64encode(BACKGROUND_PATH.read_bytes()).decode()

@st.cache_resource
def load_model(): return joblib.load(ROOT / "food_preservation_model.pkl")

@st.cache_data
def load_data(): return pd.read_csv(ROOT / "food_preservation_dataset.csv")

@st.cache_data
def load_metrics(): return pd.read_csv(ROOT / "model_metrics.csv")

def options(df):
    return {c: sorted(df[c].unique()) for c in ["food_type", "packaging_type", "storage_condition", "preservative_used"]}

def predict(model, values):
    row = pd.DataFrame([values])[FEATURES]
    p = model.predict_proba(row)[0]; classes = list(model.classes_)
    yes = float(p[classes.index(1)])
    return int(model.predict(row)[0]), yes, 1 - yes


def chart(fig, key, height=340):
    """Render a consistently sized, neutral Plotly panel."""
    fig.update_layout(height=height, paper_bgcolor="#FFFFFF", plot_bgcolor="#FCFDFC")
    st.plotly_chart(fig, config={"displaylogo": False, "responsive": True}, width="stretch", key=key)

st.markdown(f"""<style>
.stApp {{background:linear-gradient(rgba(247,248,245,.42),rgba(247,248,245,.48)),url('{background_url()}');background-size:cover;background-position:center;background-attachment:fixed;color:#1F2933;font-family:Inter,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;}}
.block-container {{max-width:1320px;padding-top:1.25rem;padding-bottom:3rem;}}
.hero {{background:rgba(255,255,255,.95);background-image:linear-gradient(90deg,rgba(255,255,255,.98),rgba(255,255,255,.58)),url('{background_url()}');background-size:cover;background-position:center;padding:2.15rem 2rem;border:1px solid #E5E7E3;border-left:5px solid #2F6B4F;border-radius:10px;box-shadow:0 2px 6px rgba(31,41,51,.05);}}
.hero h1 {{margin:0;font-size:2.1rem;color:#1F2933!important;font-weight:750}} .hero p {{margin:.3rem 0 0;color:#667085!important}}
.eyebrow {{display:inline-block;margin-bottom:.7rem;padding:.3rem .65rem;border-radius:99px;background:#E8F1EB;color:#2F6B4F!important;font-size:.72rem;font-weight:750;letter-spacing:.08em;text-transform:uppercase;}}
.guide-card {{height:100%;padding:1.1rem 1.15rem;background:#FFFFFF;border:1px solid #E5E7E3;border-radius:9px;box-shadow:0 1px 3px rgba(31,41,51,.04);}}
.guide-card h3 {{margin:.15rem 0 .5rem;font-size:1rem;color:#2F6B4F!important;}}
.guide-card p {{margin:0;color:#667085!important;font-size:.9rem;line-height:1.55;}}
[data-testid="stMetric"] {{background:#FFFFFF;border:1px solid #E5E7E3;border-top:3px solid #7FA88A;border-radius:8px;padding:1rem;box-shadow:0 1px 3px rgba(31,41,51,.04);}}
[data-testid="stMetricLabel"] {{color:#667085!important;font-size:.72rem;font-weight:700;letter-spacing:.06em;}} [data-testid="stMetricValue"] {{color:#1F2933!important;font-weight:750;}}
label,h2,h3,p,.stMarkdown {{color:#1F2933!important;}} [data-testid="stPlotlyChart"] {{background:#FFFFFF;border:1px solid #E5E7E3;border-radius:8px;padding:.3rem;box-shadow:none;}}
[data-testid="stForm"] {{background:#FFFFFF;border:1px solid #E5E7E3;border-radius:8px;padding:1.2rem;}}
[data-testid="stSidebar"] {{background:#FFFFFF;border-right:1px solid #E5E7E3;}} [data-testid="stSidebar"] * {{color:#1F2933!important;}}
[data-testid="stSidebar"] [data-baseweb="radio"] {{padding:.5rem .35rem;border-radius:6px;}} [data-testid="stSidebar"] [data-baseweb="radio"]:has(input:checked) {{background:#E8F1EB;}}
.stButton>button {{background:#2F6B4F;color:#FFFFFF!important;border:1px solid #2F6B4F;border-radius:6px;font-weight:650;}} .stButton>button:hover {{background:#25583f;border-color:#25583f;}}
</style>""", unsafe_allow_html=True)

data = load_data(); model = load_model(); metrics = load_metrics(); opts = options(data)
st.markdown("""<div class="hero"><span class="eyebrow">Interactive project</span><h1>Food Preservation Safety</h1><p>Explore storage patterns, compare model performance, and estimate preservation outcomes.</p></div>""", unsafe_allow_html=True)
with st.sidebar:
    st.markdown("## Food Safety")
    st.caption("PRESERVATION PROJECT")
    st.divider()
    page = st.radio("Navigation", ["Dashboard", "Predict", "Analytics", "Model Performance", "Project Guide"], label_visibility="collapsed")

if page == "Dashboard":
    k = get_kpis(data); st.subheader("Executive overview")
    cols = st.columns(4)
    metric_help = ["Simulated storage records in the project dataset.", "Distinct food categories represented.", "Share of records labeled Preserved.", "Records labeled Not Preserved."]
    for c, label, value, help_text in zip(cols, ["Records", "Food types", "Preservation rate", "Not preserved"],
        [f"{k['total_records']:,}", k['food_types'], f"{k['preservation_rate']:.1%}", f"{k['not_preserved_count']:,}"], metric_help): c.metric(label, value, help=help_text)
    primary, secondary = st.columns([1.25, 1], gap="large")
    with primary:
        chart(plot_target_distribution(data), "dash_target", height=510)
    with secondary:
        chart(plot_food_type_distribution(data), "dash_food", height=245)
        chart(plot_preservation_rate(data, "storage_condition", "Preservation rate by storage condition"), "dash_storage", height=245)
    st.subheader("Key risk signals")
    l,r = st.columns(2, gap="large")
    with l: st.plotly_chart(plot_food_type_preservation(data), config={"displaylogo":False}, width="stretch", key="dash_food_outcome")
    with r: st.plotly_chart(plot_score_risk_categories(data), config={"displaylogo":False}, width="stretch", key="dash_risk")

elif page == "Predict":
    st.subheader("Predict preservation")
    st.caption(f"Enter storage conditions to get an estimate from the saved {model.named_steps['model'].__class__.__name__} pipeline.")
    with st.form("prediction_form"):
        left,right = st.columns(2, gap="large")
        with left:
            food_type=st.selectbox("Food type",opts["food_type"],index=opts["food_type"].index("Milk"))
            temperature_c=st.number_input("Temperature (°C)",-30.0,50.0,4.0,.1)
            humidity_percent=st.number_input("Humidity (%)",0.0,100.0,60.0,.1)
            storage_days=st.number_input("Storage duration (days)",0.0,1000.0,2.0,.1)
            ph_level=st.number_input("pH level",0.0,14.0,6.6,.1)
        with right:
            packaging_type=st.selectbox("Packaging type",opts["packaging_type"],index=opts["packaging_type"].index("Sealed"))
            storage_condition=st.selectbox("Storage condition",opts["storage_condition"],index=opts["storage_condition"].index("Refrigerated"))
            microbial_count_cfu=st.number_input("Microbial count (CFU/g; CFU/mL for milk)",0,3_200_000_000,300,100)
            preservative_used=st.selectbox("Preservative used",opts["preservative_used"],index=opts["preservative_used"].index("No"))
            expiration_days_remaining=st.number_input("Expiration days remaining",-1000,1000,5,1)
        submitted=st.form_submit_button("Predict preservation",type="primary",width="stretch")
    if submitted:
        values=locals().copy(); values={key:values[key] for key in FEATURES}
        label,yes,no=predict(model,values)
        st.success("Estimated outcome: PRESERVED" if label else "Estimated outcome: NOT PRESERVED")
        x,y=st.columns(2); x.metric("Preserved",f"{yes:.1%}"); y.metric("Not preserved",f"{no:.1%}")
        x.progress(yes, text="Preserved probability")
        y.progress(no, text="Not preserved probability")

elif page == "Analytics":
    st.subheader("Storage and risk analytics")
    first,second = st.tabs(["Conditions", "Relationships"])
    with first:
        a,b=st.columns(2,gap="large")
        with a: st.plotly_chart(plot_distribution(data,"temperature_c"),config={"displaylogo":False},width="stretch",key="an_temp_dist")
        with b: st.plotly_chart(plot_box_by_target(data,"temperature_c"),config={"displaylogo":False},width="stretch",key="an_temp_box")
        a,b=st.columns(2,gap="large")
        with a: st.plotly_chart(plot_distribution(data,"humidity_percent"),config={"displaylogo":False},width="stretch",key="an_hum")
        with b: st.plotly_chart(plot_preservation_rate(data,"packaging_type"),config={"displaylogo":False},width="stretch",key="an_pack")
    with second:
        a,b=st.columns(2,gap="large")
        with a: st.plotly_chart(plot_microbial_scatter(data),config={"displaylogo":False},width="stretch",key="an_micro")
        with b: st.plotly_chart(plot_correlation_heatmap(data),config={"displaylogo":False},width="stretch",key="an_corr")

elif page == "Model Performance":
    selected = metrics.loc[metrics.Selected.astype(str).str.lower().eq("true")].iloc[0]
    st.subheader("Model performance")
    st.success(f"Selected model: {selected['Model']} — chosen using training-only cross-validation, prioritizing F1 for the Not Preserved class.")
    st.caption("The comparison table reports scores on a held-out test set. Cross-validation scores informed model selection.")
    display = metrics[["Model","Accuracy","F1 (Not Preserved)","Recall (Not Preserved)","ROC-AUC","Selected"]].copy()
    st.dataframe(display.style.format({"Accuracy":"{:.2%}","F1 (Not Preserved)":"{:.2%}","Recall (Not Preserved)":"{:.2%}","ROC-AUC":"{:.2%}"}),hide_index=True,width="stretch")
    a,b=st.columns(2,gap="large")
    with a: st.plotly_chart(plot_model_comparison(metrics),config={"displaylogo":False},width="stretch",key="model_comparison")
    with b: st.plotly_chart(plot_feature_importance(model),config={"displaylogo":False},width="stretch",key="model_importance")
    with st.expander("Metric definitions"):
        st.markdown("**Accuracy** is the share of all correct predictions. **F1 (Not Preserved)** balances precision and recall for that class. **Recall (Not Preserved)** measures how many Not Preserved records the model identifies. **ROC-AUC** summarizes ranking quality across classification thresholds.")

else:
    st.subheader("Project guide")
    st.markdown("This project brings together a generated dataset, exploratory analytics, a trained classifier, and an interactive prediction form. The included dataset contains **200,000 simulated records** across eight food categories.")
    stages = st.columns(3, gap="medium")
    cards = [
        ("01 · Generate", "Storage conditions and food attributes are combined into simulated records with an analytical preservation score and a Preserved / Not Preserved outcome."),
        ("02 · Explore", "The dashboard summarizes outcomes and compares preservation patterns by food type, storage condition, packaging, temperature, humidity, and microbial count."),
        ("03 · Predict", "A saved preprocessing-and-classification pipeline takes the ten form inputs and returns an estimated class with probabilities."),
    ]
    for column, (title, body) in zip(stages, cards):
        with column:
            st.markdown(f'<div class="guide-card"><h3>{title}</h3><p>{body}</p></div>', unsafe_allow_html=True)
    st.markdown("### What goes into a prediction")
    left, right = st.columns(2, gap="large")
    with left:
        st.markdown("**Food and storage**")
        st.markdown("Food type, temperature, humidity, storage duration, pH, packaging, storage condition, preservative use, and days remaining on the stated shelf life.")
    with right:
        st.markdown("**Microbial count**")
        st.markdown("The form also accepts a microbial count in CFU/g, or CFU/mL for milk. The saved pipeline encodes the category fields and scales numeric fields before classification.")
    st.markdown("### How to read the results")
    st.markdown("The dashboard’s preservation rate and class counts summarize the dataset labels. Model scores compare predictions with held-out labels. The selected model is chosen from training-only cross-validation, prioritizing F1 for Not Preserved records. Feature importance ranks the model inputs it relied on most.")
    st.info("The dataset is generated from documented simulation assumptions in `generate_dataset.py`; it is not collected from real storage batches.")
