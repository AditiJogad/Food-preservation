"""Reusable Plotly figures for the existing Streamlit application.

Pass the loaded dataset to the functions, then render with st.plotly_chart(fig).
All figures use actual aggregations; expensive plots reproducibly sample 5,000 rows.
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from analytics import add_categories, preservation_rate

PASTELS = ["#2F6B4F", "#7FA88A", "#D98B4A", "#6E9BB8", "#D6B85A", "#C96A5B"]
TEMPLATE = go.layout.Template(layout=go.Layout(
    font=dict(family='Inter, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif', color="#1F2933"),
    paper_bgcolor="#FFFFFF", plot_bgcolor="#FCFDFC", colorway=PASTELS,
    margin=dict(l=46, r=24, t=56, b=42), hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#1F2933")),
    xaxis=dict(gridcolor="#EDF0EB", linecolor="#DDE3DC", zeroline=False),
    yaxis=dict(gridcolor="#EDF0EB", linecolor="#DDE3DC", zeroline=False)))
CONFIG = {"displaylogo": False, "responsive": True}


def _style(fig, title):
    return fig.update_layout(
        template=TEMPLATE,
        title=dict(text=title, font=dict(color="#1F2933", size=17, family='Inter, system-ui, sans-serif'), x=.02),
        font=dict(color="#1F2933", family='Inter, system-ui, sans-serif'),
        paper_bgcolor="#FFFFFF", plot_bgcolor="#FFFFFF",
        legend=dict(title_text="", font=dict(color="#1F2933"), bgcolor="rgba(255,255,255,0)"),
        hoverlabel=dict(bgcolor="#FFFFFF", font=dict(color="#1F2933")),
        hovermode="closest",
    ).update_xaxes(
        title_font=dict(color="#1F2933"), tickfont=dict(color="#667085"),
        gridcolor="#E5E7E3", linecolor="#D0D7D2", zeroline=False,
    ).update_yaxes(
        title_font=dict(color="#1F2933"), tickfont=dict(color="#667085"),
        gridcolor="#E5E7E3", linecolor="#D0D7D2", zeroline=False,
    )


def plot_target_distribution(df, donut=True):
    counts = df.preserved.value_counts().rename_axis("preserved").reset_index(name="count")
    fig = px.pie(counts, names="preserved", values="count", hole=.48 if donut else 0,
        color="preserved", color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]})
    return _style(fig, "Preserved vs Not Preserved").update_traces(textinfo="label+percent+value")


def plot_target_bar(df):
    counts = df.preserved.value_counts().rename_axis("preserved").reset_index(name="count")
    counts["percent"] = counts["count"] / len(df) * 100
    fig = px.bar(counts, x="preserved", y="count", color="preserved", text=counts.percent.map("{:.1f}%".format),
        color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]})
    return _style(fig, "Target distribution").update_traces(textposition="outside")


def plot_food_type_distribution(df):
    counts = df.food_type.value_counts().rename_axis("food_type").reset_index(name="count").sort_values("count")
    return _style(px.bar(counts, x="count", y="food_type", orientation="h", color="food_type", color_discrete_sequence=PASTELS), "Total records by food type")


def plot_food_type_preservation(df):
    counts = df.groupby(["food_type", "preserved"], observed=True).size().reset_index(name="count")
    return _style(px.bar(counts, x="food_type", y="count", color="preserved", barmode="group",
        color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]}), "Food type vs preservation")


def plot_preservation_rate(df, group, title=None):
    summary = preservation_rate(df, group).sort_values("preservation_rate")
    fig = px.bar(summary, x="preservation_rate", y=group, orientation="h", text=summary.preservation_rate.map("{:.1f}%".format), color_discrete_sequence=[PASTELS[0]])
    return _style(fig, title or f"Preservation rate by {group.replace('_', ' ')}").update_layout(xaxis_title="Preservation rate (%)").update_traces(textposition="outside")


def plot_distribution(df, column, title=None, log_x=False):
    fig = px.histogram(df, x=column, nbins=45, color="preserved", barmode="overlay", opacity=.72,
        color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]}, log_x=log_x)
    return _style(fig, title or f"{column.replace('_', ' ').title()} distribution")


def plot_box_by_target(df, column, title=None, log_y=False):
    fig = px.box(df, x="preserved", y=column, color="preserved", points=False,
        color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]}, log_y=log_y)
    return _style(fig, title or f"{column.replace('_', ' ').title()} vs preservation")


def plot_microbial_scatter(df, sample_size=5000):
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    fig = px.scatter(sample, x="storage_days", y="microbial_count_cfu", color="preserved", log_y=True,
        opacity=.55, color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]},
        hover_data=["food_type", "storage_condition", "temperature_c"])
    return _style(fig, "Microbial count vs storage days (5,000-row sample)")


def plot_expiration_distribution(df):
    categorized = add_categories(df)
    counts = categorized.expiration_status.value_counts().reindex(["Expired", "Expires Today", "Expiring Soon", "Valid"]).reset_index()
    counts.columns = ["expiration_status", "count"]
    return _style(px.bar(counts, x="expiration_status", y="count", color="expiration_status", color_discrete_sequence=PASTELS), "Expiration status distribution")


def plot_packaging_distribution(df):
    counts = df.packaging_type.value_counts().rename_axis("packaging_type").reset_index(name="count")
    return _style(px.bar(counts, x="packaging_type", y="count", color="packaging_type", color_discrete_sequence=PASTELS), "Records by packaging type")


def plot_storage_condition_distribution(df):
    counts = df.storage_condition.value_counts().rename_axis("storage_condition").reset_index(name="count")
    return _style(px.bar(counts, x="storage_condition", y="count", color="storage_condition", color_discrete_sequence=PASTELS), "Storage condition distribution")


def plot_preservative_donut(df):
    counts = df.preservative_used.value_counts().rename_axis("preservative_used").reset_index(name="count")
    return _style(px.pie(counts, names="preservative_used", values="count", hole=.48, color_discrete_sequence=PASTELS), "Preservative use")


def plot_food_type_ph(df):
    return _style(px.box(df, x="food_type", y="ph_level", color="food_type", points=False, color_discrete_sequence=PASTELS), "Food type vs pH")


def plot_score_risk_categories(df):
    categorized = add_categories(df)
    counts = categorized.risk_category.value_counts().reindex(["Low", "Moderate", "High", "Critical"]).reset_index()
    counts.columns = ["risk_category", "count"]
    return _style(px.bar(counts, x="risk_category", y="count", color="risk_category", color_discrete_sequence=PASTELS), "Preservation score risk categories")


def plot_correlation_heatmap(df):
    columns = ["temperature_c", "humidity_percent", "storage_days", "ph_level", "microbial_count_cfu", "expiration_days_remaining", "preservation_score"]
    corr = df[columns].corr().round(3)
    fig = px.imshow(corr, text_auto=True, color_continuous_scale=[[0, "#8FB9E3"], [.5, "#FFFFFF"], [1, "#F2A6B3"]], zmin=-1, zmax=1, aspect="auto")
    return _style(fig, "Numerical correlation heatmap")


def plot_scatter_matrix(df, sample_size=5000):
    columns = ["temperature_c", "storage_days", "microbial_count_cfu", "expiration_days_remaining", "preservation_score"]
    sample = df.sample(min(sample_size, len(df)), random_state=42)
    fig = px.scatter_matrix(sample, dimensions=columns, color="preserved", symbol="preserved",
        color_discrete_map={"Yes": PASTELS[3], "No": PASTELS[1]}, opacity=.45)
    return _style(fig, "Scatter matrix (5,000-row sample)").update_traces(diagonal_visible=False)


def plot_model_comparison(metrics):
    long = metrics.melt(id_vars="Model", value_vars=["Accuracy", "Precision (Preserved)", "Recall (Preserved)", "F1 (Preserved)", "ROC-AUC"], var_name="Metric", value_name="Score")
    return _style(px.bar(long, x="Model", y="Score", color="Metric", barmode="group", color_discrete_sequence=PASTELS, hover_data={"Score": ":.3f"}), "Model comparison")


def plot_model_radar(metrics):
    metrics_cols = ["Accuracy", "Precision (Preserved)", "Recall (Preserved)", "F1 (Preserved)", "ROC-AUC"]
    fig = go.Figure()
    for _, row in metrics.iterrows():
        fig.add_trace(go.Scatterpolar(r=[row[x] for x in metrics_cols], theta=metrics_cols, fill="toself", name=row.Model, opacity=.5))
    return _style(fig, "Model comparison radar").update_layout(polar=dict(radialaxis=dict(range=[0, 1], visible=True)))


def plot_feature_importance(model, top_n=15):
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        raise ValueError("Selected model does not expose feature_importances_.")
    values = pd.DataFrame({"feature": model.named_steps["preprocessor"].get_feature_names_out(), "importance": estimator.feature_importances_}).nlargest(top_n, "importance").sort_values("importance")
    fig = px.bar(values, x="importance", y="feature", orientation="h", color_discrete_sequence=[PASTELS[2]])
    return _style(fig, "Top model features")


def plot_confusion_matrix(matrix):
    fig = px.imshow(matrix, text_auto=True, x=["Predicted Not Preserved", "Predicted Preserved"], y=["Actual Not Preserved", "Actual Preserved"], color_continuous_scale=[[0, "#FFFFFF"], [1, "#B8A1D9"]])
    return _style(fig, "Selected model confusion matrix")


def plot_roc_curves(curves):
    """curves maps model name to (false_positive_rate, true_positive_rate, auc)."""
    fig = go.Figure()
    for index, (name, (fpr, tpr, auc)) in enumerate(curves.items()): fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC {auc:.3f})", line=dict(color=PASTELS[index % len(PASTELS)])))
    fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Chance", line=dict(color="#87939A", dash="dash")))
    return _style(fig, "ROC curves").update_layout(xaxis_title="False positive rate", yaxis_title="True positive rate")


def plot_precision_recall_curves(curves):
    """curves maps model name to (recall, precision)."""
    fig = go.Figure()
    for index, (name, (recall, precision)) in enumerate(curves.items()): fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name, line=dict(color=PASTELS[index % len(PASTELS)])))
    return _style(fig, "Precision–recall curves").update_layout(xaxis_title="Recall", yaxis_title="Precision")
