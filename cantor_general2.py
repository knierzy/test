import io
import itertools
import math
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Cantor Grids")

st.title("Cantor Grids – Four-Parameter Compositional Visualization")
st.caption(
    "Define four compositional parameters, create subgroup fields from parameter ranges, "
    "and upload your own four-parameter dataset for visualization."
)

# ============================================================
# Helper functions
# ============================================================

def normalize_to_100_lrm(values):
    """Normalize four non-negative values to integer percentages summing to 100."""
    arr = np.asarray(values, dtype=float)

    if np.any(~np.isfinite(arr)):
        raise ValueError("All four parameter values must be finite numbers.")
    if np.any(arr < 0):
        raise ValueError("Parameter values must be non-negative.")
    if arr.sum() <= 0:
        raise ValueError("The sum of the four parameter values must be > 0.")

    scaled = arr / arr.sum() * 100.0
    ints = np.floor(scaled).astype(int)
    remainder = scaled - ints
    missing = 100 - ints.sum()

    if missing > 0:
        order = np.argsort(-remainder)
        for i in range(missing):
            ints[order[i % len(order)]] += 1
    elif missing < 0:
        order = np.argsort(remainder)
        for i in range(-missing):
            idx = order[i % len(order)]
            if ints[idx] > 0:
                ints[idx] -= 1

    return ints


# Each valid integer composition maps to one unique grid point.
# We retain the same logic as in the original Cantor-grid implementation:
# AB determines the large row, B gives the within-row vertical position,
# and C gives the within-row horizontal position.
RECTANGLES = [
    (0, 100, "AB99"), (100, 99, "AB98"), (199, 98, "AB97"), (297, 97, "AB96"), (394, 96, "AB95"),
    (490, 95, "AB94"), (585, 94, "AB93"), (679, 93, "AB92"), (772, 92, "AB91"), (864, 91, "AB90"),
    (955, 90, "AB89"), (1045, 89, "AB88"), (1134, 88, "AB87"), (1222, 87, "AB86"), (1309, 86, "AB85"),
    (1395, 85, "AB84"), (1480, 84, "AB83"), (1564, 83, "AB82"), (1647, 82, "AB81"), (1729, 81, "AB80"),
    (1810, 80, "AB79"), (1890, 79, "AB78"), (1969, 78, "AB77"), (2047, 77, "AB76"), (2124, 76, "AB75"),
    (2200, 75, "AB74"), (2275, 74, "AB73"), (2349, 73, "AB72"), (2422, 72, "AB71"), (2494, 71, "AB70"),
    (2565, 70, "AB69"), (2635, 69, "AB68"), (2704, 68, "AB67"), (2772, 67, "AB66"), (2839, 66, "AB65"),
    (2905, 65, "AB64"), (2970, 64, "AB63"), (3034, 63, "AB62"), (3097, 62, "AB61"), (3159, 61, "AB60"),
    (3220, 60, "AB59"), (3280, 59, "AB58"), (3339, 58, "AB57"), (3397, 57, "AB56"), (3454, 56, "AB55"),
    (3510, 55, "AB54"), (3565, 54, "AB53"), (3619, 53, "AB52"), (3672, 52, "AB51"), (3724, 51, "AB50"),
    (3775, 50, "AB49"), (3825, 49, "AB48"), (3874, 48, "AB47"), (3922, 47, "AB46"), (3969, 46, "AB45"),
    (4015, 45, "AB44"), (4060, 44, "AB43"), (4104, 43, "AB42"), (4147, 42, "AB41"), (4189, 41, "AB40"),
    (4230, 40, "AB39"), (4270, 39, "AB38"), (4309, 38, "AB37"), (4347, 37, "AB36"), (4384, 36, "AB35"),
    (4420, 35, "AB34"), (4455, 34, "AB33"), (4489, 33, "AB32"), (4522, 32, "AB31"), (4554, 31, "AB30"),
    (4585, 30, "AB29"), (4615, 29, "AB28"), (4644, 28, "AB27"), (4672, 27, "AB26"), (4699, 26, "AB25"),
    (4725, 25, "AB24"), (4750, 24, "AB23"), (4774, 23, "AB22"), (4797, 22, "AB21"), (4819, 21, "AB20"),
    (4840, 20, "AB19"), (4860, 19, "AB18"), (4879, 18, "AB17"), (4897, 17, "AB16"), (4914, 16, "AB15"),
    (4930, 15, "AB14"), (4945, 14, "AB13"), (4959, 13, "AB12"), (4972, 12, "AB11"), (4984, 11, "AB10"),
    (4995, 10, "AB9"), (5005, 9, "AB8"), (5014, 8, "AB7"), (5022, 7, "AB6"), (5029, 6, "AB5"),
    (5035, 5, "AB4"), (5040, 4, "AB3"), (5044, 3, "AB2"), (5047, 2, "AB1")
]

def cantor_xy(a, b, c, d):
    total = a + b + c + d
    if total != 100:
        return None, None

    ab = int(a + b)
    ab_index = 99 - ab

    if ab_index < 0 or ab_index >= len(RECTANGLES):
        return None, None

    start_pos, height, _ = RECTANGLES[ab_index]

    # Original script builds x=c, y=row-position+b and rotates the entire plot.
    # We directly return the final, rotated coordinates:
    x = start_pos + b + 0.5
    y = c + 0.5
    return x, y


def build_subgroup_points(ranges):
    """
    Create the constrained Cartesian product for integer compositions.
    ranges is [(Amin,Amax), (Bmin,Bmax), (Cmin,Cmax), (Dmin,Dmax)].
    """
    vals = [range(int(lo), int(hi) + 1) for lo, hi in ranges]
    out = []

    for a, b, c, d in itertools.product(*vals):
        if a + b + c + d == 100:
            x, y = cantor_xy(a, b, c, d)
            if x is not None:
                out.append((a, b, c, d, x, y))

    return pd.DataFrame(out, columns=["A", "B", "C", "D", "x", "y"])


def read_uploaded_dataset(uploaded_file, labels):
    df = pd.read_excel(uploaded_file)

    # Prefer actual parameter labels.
    wanted = list(labels)
    generic = ["A", "B", "C", "D"]

    if all(col in df.columns for col in wanted):
        mapping = {wanted[i]: generic[i] for i in range(4)}
        data = df[wanted].rename(columns=mapping).copy()
    elif all(col in df.columns for col in generic):
        data = df[generic].copy()
    elif df.shape[1] >= 4:
        data = df.iloc[:, :4].copy()
        data.columns = generic
    else:
        raise ValueError(
            "The Excel file must contain at least four parameter columns."
        )

    if "Locality" in df.columns:
        data["Locality"] = df["Locality"]
    elif df.shape[1] >= 5:
        data["Locality"] = df.iloc[:, 4].astype(str)
    else:
        data["Locality"] = [f"Sample {i+1}" for i in range(len(data))]

    # Normalize every row to integer closure = 100.
    normalized = []
    for _, row in data.iterrows():
        vals = normalize_to_100_lrm(row[generic].values)
        normalized.append(vals)

    norm = np.vstack(normalized)
    data[generic] = norm

    coords = data.apply(
        lambda r: cantor_xy(int(r["A"]), int(r["B"]), int(r["C"]), int(r["D"])),
        axis=1
    )

    data["x"] = [xy[0] for xy in coords]
    data["y"] = [xy[1] for xy in coords]

    return data


def classify_point_by_ranges(row, subgroup_defs):
    memberships = []

    for sg in subgroup_defs:
        vals = [row["A"], row["B"], row["C"], row["D"]]
        inside = all(
            sg["ranges"][i][0] <= vals[i] <= sg["ranges"][i][1]
            for i in range(4)
        )
        if inside:
            memberships.append(sg["name"])

    return ", ".join(memberships) if memberships else "Unclassified"


# ============================================================
# Parameter names
# ============================================================

st.header("1. Define parameter names")

col1, col2, col3, col4 = st.columns(4)

with col1:
    label_a = st.text_input("Parameter A", value="Sand")
with col2:
    label_b = st.text_input("Parameter B", value="Silt")
with col3:
    label_c = st.text_input("Parameter C", value="Clay")
with col4:
    label_d = st.text_input("Parameter D", value="Organic matter")

labels = [label_a.strip() or "A", label_b.strip() or "B", label_c.strip() or "C", label_d.strip() or "D"]

if len(set(labels)) < 4:
    st.error("The four parameter names must be different.")
    st.stop()

# ============================================================
# Subgroup builder
# ============================================================

st.header("2. Define subgroup fields")

if "subgroup_count" not in st.session_state:
    st.session_state.subgroup_count = 1

btn_col1, btn_col2, _ = st.columns([1, 1, 4])

with btn_col1:
    if st.button("+ Add subgroup"):
        st.session_state.subgroup_count += 1
        st.rerun()

with btn_col2:
    if st.button("− Remove last subgroup") and st.session_state.subgroup_count > 1:
        st.session_state.subgroup_count -= 1
        st.rerun()

subgroup_defs = []

for i in range(st.session_state.subgroup_count):
    with st.expander(f"Subgroup {i+1}", expanded=True):
        name = st.text_input(
            "Subgroup name",
            value=f"Subgroup {i+1}",
            key=f"sg_name_{i}"
        )

        ranges = []
        for j, label in enumerate(labels):
            c1, c2 = st.columns(2)
            with c1:
                lo = st.number_input(
                    f"{label} minimum",
                    min_value=0,
                    max_value=100,
                    value=0,
                    step=1,
                    key=f"sg_{i}_{j}_lo"
                )
            with c2:
                hi = st.number_input(
                    f"{label} maximum",
                    min_value=0,
                    max_value=100,
                    value=100,
                    step=1,
                    key=f"sg_{i}_{j}_hi"
                )

            if lo > hi:
                st.error(f"{label}: minimum must not exceed maximum.")
            ranges.append((int(lo), int(hi)))

        subgroup_defs.append({
            "name": name.strip() or f"Subgroup {i+1}",
            "ranges": ranges
        })

# ============================================================
# Build subgroup Cartesian products
# ============================================================

st.header("3. Generate subgroup fields")

generate = st.button("Generate subgroup fields")

if generate:
    st.session_state["generated_subgroups"] = []

    for sg in subgroup_defs:
        pts = build_subgroup_points(sg["ranges"])
        st.session_state["generated_subgroups"].append({
            "name": sg["name"],
            "ranges": sg["ranges"],
            "points": pts
        })

if "generated_subgroups" in st.session_state:
    generated_subgroups = st.session_state["generated_subgroups"]

    summary_rows = []
    for sg in generated_subgroups:
        summary_rows.append({
            "Subgroup": sg["name"],
            "Valid integer compositions": len(sg["points"])
        })

    st.dataframe(pd.DataFrame(summary_rows), use_container_width=True)

    empty_groups = [sg["name"] for sg in generated_subgroups if sg["points"].empty]
    if empty_groups:
        st.warning(
            "No valid compositions summing to 100 were generated for: "
            + ", ".join(empty_groups)
        )
else:
    generated_subgroups = []

# ============================================================
# Upload data
# ============================================================

st.header("4. Upload dataset")

st.markdown(
    f"""
Your Excel file may use either:

- **generic columns:** `A`, `B`, `C`, `D`, `Locality`
- **actual parameter names:** `{labels[0]}`, `{labels[1]}`, `{labels[2]}`, `{labels[3]}`, `Locality`

If no `Locality` column is present, sample names are created automatically.

Uploaded rows are normalized by largest-remainder integer closure to 100%.
"""
)

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])

# ============================================================
# Plot controls
# ============================================================

st.header("5. Plot settings")

ctrl1, ctrl2 = st.columns(2)

with ctrl1:
    point_size = st.slider(
        "Point size",
        min_value=5,
        max_value=40,
        value=18,
        step=1
    )

with ctrl2:
    colorscale = st.selectbox(
        "Point color scale",
        ["Plasma", "Viridis", "Turbo", "Inferno", "Cividis", "RdYlBu"]
    )

show_subgroup_points = st.checkbox(
    "Show all valid subgroup compositions",
    value=True
)

# ============================================================
# Create plot
# ============================================================

if uploaded_file is not None:
    try:
        df = read_uploaded_dataset(uploaded_file, labels)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    # Recompute subgroup definitions from current widgets for classification.
    current_subgroups = subgroup_defs

    df["Subgroup"] = df.apply(
        lambda row: classify_point_by_ranges(row, current_subgroups),
        axis=1
    )

    fig = go.Figure()

    # Plot subgroup fields as clouds of valid compositions.
    if show_subgroup_points and generated_subgroups:
        for sg in generated_subgroups:
            pts = sg["points"]
            if pts.empty:
                continue

            fig.add_trace(
                go.Scattergl(
                    x=pts["x"],
                    y=pts["y"],
                    mode="markers",
                    marker=dict(
                        size=5,
                        opacity=0.22
                    ),
                    name=sg["name"],
                    hovertemplate=(
                        f"<b>{sg['name']}</b><br>"
                        + f"{labels[0]}: %{{customdata[0]}}%<br>"
                        + f"{labels[1]}: %{{customdata[1]}}%<br>"
                        + f"{labels[2]}: %{{customdata[2]}}%<br>"
                        + f"{labels[3]}: %{{customdata[3]}}%"
                        + "<extra></extra>"
                    ),
                    customdata=pts[["A", "B", "C", "D"]].to_numpy()
                )
            )

    ratio = df["A"] / (df["A"] + df["B"]).replace(0, np.nan)
    ratio = ratio.fillna(0)

    hover_text = [
        (
            f"<b>{loc}</b><br>"
            f"{labels[0]}: {a:.0f}%<br>"
            f"{labels[1]}: {b:.0f}%<br>"
            f"{labels[2]}: {c:.0f}%<br>"
            f"{labels[3]}: {d:.0f}%<br>"
            f"Subgroup: {sg}"
        )
        for loc, a, b, c, d, sg
        in zip(
            df["Locality"], df["A"], df["B"], df["C"], df["D"], df["Subgroup"]
        )
    ]

    fig.add_trace(
        go.Scattergl(
            x=df["x"],
            y=df["y"],
            mode="markers",
            marker=dict(
                size=point_size,
                color=ratio,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                line=dict(width=1.5),
                colorbar=dict(
                    title=f"{labels[0]} / ({labels[0]} + {labels[1]})"
                )
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Uploaded samples"
        )
    )

    fig.update_layout(
        template="plotly_white",
        height=900,
        xaxis=dict(
            title=f"Cantor-grid position based on {labels[0]} + {labels[1]} and {labels[1]}",
            showgrid=False,
            zeroline=False
        ),
        yaxis=dict(
            title=f"{labels[2]} (%)",
            range=[0, 101],
            dtick=10,
            showgrid=True,
            zeroline=False
        ),
        legend=dict(
            title="Subgroups",
            orientation="v"
        ),
        margin=dict(l=60, r=80, t=40, b=80)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normalized uploaded data")
    display_df = df[["Locality", "A", "B", "C", "D", "Subgroup"]].copy()
    display_df = display_df.rename(
        columns={
            "A": labels[0],
            "B": labels[1],
            "C": labels[2],
            "D": labels[3]
        }
    )
    st.dataframe(display_df, use_container_width=True)

    csv_bytes = display_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download classified data (CSV)",
        data=csv_bytes,
        file_name="cantor_grids_classified_data.csv",
        mime="text/csv"
    )

else:
    st.info("Upload an Excel file to create the Cantor-grid plot.")
