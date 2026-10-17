"""Chart builders shared by the interface and the exported figures (docs/figures)."""

from __future__ import annotations

import altair as alt
import pandas as pd

# Fixed entity -> colour mapping (reference palette slots 1-2; the constant baseline is a neutral reference)
COLORS = {"Model B (no text)": "#2a78d6", "Model C (with text)": "#eb6834", "Constant baseline": "#7a7d82"}
INK, MUTED, RULE, ACCENT, SURFACE = "#1b1f24", "#5d6571", "#d9d5cb", "#1f4e79", "#f7f6f2"
ORDER_KEYS = ["C vs B", "C vs B (company clusters)", "C vs B (uncalibrated)", "B+length vs B", "B+dictionary vs B",
              "B+paragraph_change vs B", "B vs A", "A vs constant", "C vs constant", "boosting C vs boosting B"]


def style(chart, background: str | None = None):
    c = chart.configure_axis(labelColor=MUTED, titleColor=MUTED, gridColor="#ebe8e1", domainColor=RULE, tickColor=RULE,
                             labelFontSize=11, titleFontSize=11, titleFontWeight="normal") \
        .configure_view(strokeWidth=0).configure_legend(labelColor=INK, titleColor=MUTED, orient="top")
    return c.configure(background=background) if background else c


def paired_differences_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for k, v in metrics["paired_differences"].items():
        b = v["brier"]
        rows.append({"comparison": k.replace("logistic_l2:", "").replace("hgb_shallow:", "boosting ")
                     .replace("constant:constant", "constant"),
                     "estimate": b["estimate"], "lo": b["ci95"][0], "hi": b["ci95"][1],
                     "primary": k == "logistic_l2:C vs logistic_l2:B", "cluster": v.get("cluster"),
                     "ap": v.get("average_precision", {}).get("estimate")})
    df = pd.DataFrame(rows)
    df["_o"] = df["comparison"].map({k: i for i, k in enumerate(ORDER_KEYS)}).fillna(99)
    return df.sort_values("_o").drop(columns="_o").reset_index(drop=True)


def forest_chart(metrics: dict):
    ddf = paired_differences_frame(metrics)
    base = alt.Chart(ddf).encode(y=alt.Y("comparison:N", sort=list(ddf["comparison"]), title=None,
                                         axis=alt.Axis(labelLimit=320)))
    whisk = base.mark_rule(strokeWidth=2, color=MUTED).encode(
        x=alt.X("lo:Q", title="ΔBrier (first model − second; negative favours the first)",
                axis=alt.Axis(format="+.3f", tickCount=8)), x2="hi:Q")
    dots = base.mark_circle(size=90, opacity=1).encode(
        x="estimate:Q", color=alt.condition("datum.primary", alt.value("#eb6834"), alt.value(ACCENT)),
        tooltip=[alt.Tooltip("comparison:N"), alt.Tooltip("estimate:Q", format="+.4f"),
                 alt.Tooltip("lo:Q", title="95% low", format="+.4f"), alt.Tooltip("hi:Q", title="95% high", format="+.4f"),
                 alt.Tooltip("cluster:N", title="bootstrap clusters")])
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color=INK, strokeDash=[3, 3]).encode(x="x:Q")
    return (zero + whisk + dots).properties(height=34 * len(ddf) + 40)


def calibration_frame(metrics: dict) -> pd.DataFrame:
    rows = []
    for k, lab in [("logistic_l2:B", "Model B (no text)"), ("logistic_l2:C", "Model C (with text)")]:
        for b in metrics["models"][k]["calibration_bins"]:
            rows.append({"model": lab, **b})
    return pd.DataFrame(rows)


def calibration_chart(metrics: dict):
    cdf = calibration_frame(metrics)
    top = max(0.7, float(cdf[["mean_predicted", "observed_rate"]].max().max()) + 0.05)
    diag = alt.Chart(pd.DataFrame({"x": [0, top], "y": [0, top]})).mark_line(color=RULE, strokeDash=[4, 4]).encode(x="x:Q", y="y:Q")
    sc = alt.Scale(domain=["Model B (no text)", "Model C (with text)"], range=[COLORS["Model B (no text)"], COLORS["Model C (with text)"]])
    lines = alt.Chart(cdf).mark_line(strokeWidth=2).encode(
        x=alt.X("mean_predicted:Q", title="Mean predicted probability in bin", scale=alt.Scale(domain=[0, top])),
        y=alt.Y("observed_rate:Q", title="Observed share of large moves", scale=alt.Scale(domain=[0, top])),
        color=alt.Color("model:N", scale=sc, title=None))
    pts = alt.Chart(cdf).mark_circle(opacity=1, stroke=SURFACE, strokeWidth=2).encode(
        x="mean_predicted:Q", y="observed_rate:Q", color=alt.Color("model:N", scale=sc, title=None),
        size=alt.Size("n:Q", title="events in bin", scale=alt.Scale(range=[60, 260])),
        tooltip=["model:N", alt.Tooltip("mean_predicted:Q", format=".3f"), alt.Tooltip("observed_rate:Q", format=".3f"),
                 alt.Tooltip("n:Q", title="events"), alt.Tooltip("p_min:Q", format=".3f"), alt.Tooltip("p_max:Q", format=".3f")])
    return (diag + lines + pts).properties(height=360)


def yearly_frame(metrics: dict) -> pd.DataFrame:
    py = pd.DataFrame(metrics["per_year"])
    py = py[py["model"].isin(["constant:constant", "logistic_l2:B", "logistic_l2:C"])].copy()
    py["series"] = py["model"].map({"constant:constant": "Constant baseline", "logistic_l2:B": "Model B (no text)",
                                    "logistic_l2:C": "Model C (with text)"})
    return py


def yearly_chart(metrics: dict):
    py = yearly_frame(metrics)
    ysc = alt.Scale(domain=list(COLORS), range=list(COLORS.values()))
    return alt.Chart(py).mark_line(strokeWidth=2, point=alt.OverlayMarkDef(size=60, filled=True)).encode(
        x=alt.X("test_year:O", title="Test year", axis=alt.Axis(labelAngle=0)),
        y=alt.Y("brier:Q", title="Brier score (lower is better)", scale=alt.Scale(zero=False)),
        color=alt.Color("series:N", scale=ysc, title=None),
        strokeDash=alt.condition("datum.series == 'Constant baseline'", alt.value([4, 3]), alt.value([1, 0])),
        tooltip=["series:N", "test_year:O", alt.Tooltip("brier:Q", format=".4f"), alt.Tooltip("n_events:Q", title="events"),
                 alt.Tooltip("n_positive:Q", title="large moves")]).properties(height=300)


def funnel_chart(funnel: dict):
    fdf = pd.DataFrame(funnel["stages"])
    fdf["label"] = fdf["stage"].str.replace("_", " ")
    bars = alt.Chart(fdf).mark_bar(color=ACCENT, cornerRadiusEnd=3, height=18).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=240)),
        x=alt.X("remaining:Q", title="Filings remaining"),
        tooltip=[alt.Tooltip("label:N", title="stage"), "remaining:Q", "excluded:Q", alt.Tooltip("description:N")])
    labels = bars.mark_text(align="left", dx=4, color=INK, fontSize=11).encode(text="remaining:Q")
    return (bars + labels).properties(height=32 * len(fdf))
