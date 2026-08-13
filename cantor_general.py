import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(layout="wide", page_title="Cantor Grids")

st.title("Cantor Grids – Four-Parameter Compositional Visualization")
st.caption(
    "Define four compositional parameters, create subgroup fields from parameter ranges, "
    "and upload your own four-parameter dataset."
)

# ============================================================
# Core Cantor-grid geometry
# ============================================================

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

X_LABELS = {
    50: "AB99", 440: "AB95", 915: "AB90", 1350: "AB85",
    1760: "AB80", 2158: "AB75", 2540: "AB70",
    2870: "AB65", 3195: "AB60", 3480: "AB55",
    3755: "AB50", 3995: "AB45", 4209: "AB40",
    4405: "AB35", 4570: "AB30", 4830: "AB20",
    4990: "AB10", 5046: "AB01"
}

SUBGROUP_COLORS = [
    "rgba(31,119,180,0.85)",
    "rgba(255,127,14,0.85)",
    "rgba(44,160,44,0.85)",
    "rgba(214,39,40,0.85)",
    "rgba(148,103,189,0.85)",
    "rgba(140,86,75,0.85)",
    "rgba(227,119,194,0.85)",
    "rgba(127,127,127,0.85)",
    "rgba(188,189,34,0.85)",
    "rgba(23,190,207,0.85)"
]


def normalize_to_100_lrm(values):
    arr = np.asarray(values, dtype=float)
    if np.any(~np.isfinite(arr)) or np.any(arr < 0) or arr.sum() <= 0:
        raise ValueError("All parameter values must be finite, non-negative, and have a positive sum.")

    scaled = arr / arr.sum() * 100.0
    ints = np.floor(scaled).astype(int)
    rem = scaled - ints
    missing = int(100 - ints.sum())

    if missing > 0:
        for idx in np.argsort(-rem)[:missing]:
            ints[idx] += 1
    elif missing < 0:
        for idx in np.argsort(rem)[:abs(missing)]:
            if ints[idx] > 0:
                ints[idx] -= 1

    return ints


def calculate_raw_position(a, b, c, d):
    """
    Original (pre-rotation) Cantor coordinates.
    x = C
    y = row start determined by A+B, plus B.
    """
    if int(a + b + c + d) != 100:
        return None, None

    ab = int(a + b)
    ab_index = 99 - ab

    if ab_index < 0 or ab_index >= len(RECTANGLES):
        return None, None

    row_start, _, _ = RECTANGLES[ab_index]
    return float(c), float(row_start + b + 0.5)


def calculate_final_position(a, b, c, d):
    # The original garnet script rotates all traces at the end.
    raw_x, raw_y = calculate_raw_position(a, b, c, d)
    if raw_x is None:
        return None, None
    return raw_y, raw_x


def add_gray_cantor_grid(fig):
    """
    Reproduces the gray triangular Cantor-grid domain from the original garnet app.
    Coordinates are added directly in final (rotated) orientation.
    """
    for i, (row_start, height, label) in enumerate(RECTANGLES):
        width = i + 1
        gradient_steps = 5 if i >= 90 else 10

        gray_start = 200
        gray_end = 230

        for step in range(gradient_steps):
            gray = int(gray_start + (gray_end - gray_start) * (step / max(gradient_steps - 1, 1)))
            alpha = 0.8 - (0.6 * (step / max(gradient_steps - 1, 1)))
            fill = f"rgba({gray},{gray},{gray},{alpha})"

            raw_y0 = row_start + (step / gradient_steps) * height
            raw_y1 = row_start + ((step + 1) / gradient_steps) * height

            # raw rectangle was x=[0,width], y=[raw_y0,raw_y1]
            # after original rotation: x<-raw_y, y<-raw_x
            fig.add_trace(
                go.Scatter(
                    x=[raw_y0, raw_y1, raw_y1, raw_y0, raw_y0],
                    y=[0, 0, width, width, 0],
                    fill="toself",
                    mode="none",
                    fillcolor=fill,
                    hoverinfo="skip",
                    showlegend=False
                )
            )

        # Internal grid lines (the original vertical lines become horizontal after rotation)
        for raw_x in range(1, width):
            fig.add_trace(
                go.Scatter(
                    x=[row_start, row_start + height],
                    y=[raw_x, raw_x],
                    mode="lines",
                    line=dict(color="rgba(100,100,100,0.65)", width=0.8),
                    hoverinfo="skip",
                    showlegend=False
                )
            )


def build_subgroup_points(ranges):
    """
    Constrained Cartesian product:
    all integer A/B/C/D combinations inside the four ranges with A+B+C+D = 100.
    """
    (amin, amax), (bmin, bmax), (cmin, cmax), (dmin, dmax) = ranges
    rows = []

    # Faster than a full 4-D product: D follows from the closure constraint.
    for a in range(amin, amax + 1):
        for b in range(bmin, bmax + 1):
            ab = a + b
            if ab < 1 or ab > 99:
                continue
            for c in range(cmin, cmax + 1):
                d = 100 - a - b - c
                if d < dmin or d > dmax:
                    continue

                x, y = calculate_final_position(a, b, c, d)
                if x is not None:
                    rows.append((a, b, c, d, ab, x, y))

    return pd.DataFrame(rows, columns=["A", "B", "C", "D", "AB", "x", "y"])


def convex_hull_2d(points):
    """
    Monotonic-chain convex hull; avoids an additional scipy dependency.
    Returns hull vertices in order.
    """
    pts = sorted(set((float(x), float(y)) for x, y in points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0]-o[0]) * (b[1]-o[1]) - (a[1]-o[1]) * (b[0]-o[0])

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    return lower[:-1] + upper[:-1]


def rgba_with_alpha(rgba, alpha):
    parts = rgba.replace("rgba(", "").replace(")", "").split(",")
    return f"rgba({parts[0]},{parts[1]},{parts[2]},{alpha})"


def add_subgroup_fields(fig, subgroup_results):
    """
    Draw subgroup fields as colored rectangular outlines only.

    For every AB slice, the valid constrained Cartesian-product compositions
    define a rectangular subfield in the final Cantor-grid coordinates.
    No colored subgroup points are plotted.
    """
    for idx, sg in enumerate(subgroup_results):
        pts = sg["points"]
        if pts.empty:
            continue

        color = SUBGROUP_COLORS[idx % len(SUBGROUP_COLORS)]
        fill = rgba_with_alpha(color, 0.015)
        first_trace = True

        for ab, group in pts.groupby("AB"):
            x_min = float(group["x"].min()) - 0.45
            x_max = float(group["x"].max()) + 0.45
            y_min = float(group["y"].min()) - 0.45
            y_max = float(group["y"].max()) + 0.45

            # Colored rectangular subfield, matching the garnet-style display.
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_min, x_max, x_max, x_min],
                    y=[y_min, y_max, y_max, y_min, y_min],
                    mode="lines",
                    line=dict(color=color, width=2.4),
                    fill="toself",
                    fillcolor=fill,
                    name=sg["name"],
                    legendgroup=sg["name"],
                    showlegend=first_trace,
                    hovertemplate=(
                        f"<b>{sg['name']}</b><br>"
                        f"AB = {int(ab)}<br>"
                        f"A: {int(group['A'].min())}–{int(group['A'].max())}%<br>"
                        f"B: {int(group['B'].min())}–{int(group['B'].max())}%<br>"
                        f"C: {int(group['C'].min())}–{int(group['C'].max())}%<br>"
                        f"D: {int(group['D'].min())}–{int(group['D'].max())}%"
                        "<extra></extra>"
                    )
                )
            )
            first_trace = False


def read_uploaded_dataset(uploaded_file, labels):
    df = pd.read_excel(uploaded_file)
    generic = ["A", "B", "C", "D"]

    if all(label in df.columns for label in labels):
        data = df[labels].copy()
        data.columns = generic
    elif all(col in df.columns for col in generic):
        data = df[generic].copy()
    elif df.shape[1] >= 4:
        data = df.iloc[:, :4].copy()
        data.columns = generic
    else:
        raise ValueError("The Excel file must contain at least four parameter columns.")

    if "Locality" in df.columns:
        data["Locality"] = df["Locality"].astype(str)
    elif df.shape[1] >= 5:
        data["Locality"] = df.iloc[:, 4].astype(str)
    else:
        data["Locality"] = [f"Sample {i+1}" for i in range(len(data))]

    norm = np.vstack([normalize_to_100_lrm(row) for row in data[generic].to_numpy()])
    data[generic] = norm

    coords = [calculate_final_position(*row) for row in data[generic].to_numpy()]
    data["x"] = [p[0] for p in coords]
    data["y"] = [p[1] for p in coords]

    return data


def classify_by_ranges(row, subgroup_defs):
    hits = []
    vals = [row["A"], row["B"], row["C"], row["D"]]

    for sg in subgroup_defs:
        if all(sg["ranges"][i][0] <= vals[i] <= sg["ranges"][i][1] for i in range(4)):
            hits.append(sg["name"])

    return ", ".join(hits) if hits else "Unclassified"


# ============================================================
# 1. Parameter names
# ============================================================

st.header("1. Define parameter names")

c1, c2, c3, c4 = st.columns(4)
with c1:
    label_a = st.text_input("Parameter A", "Sand")
with c2:
    label_b = st.text_input("Parameter B", "Silt")
with c3:
    label_c = st.text_input("Parameter C", "Clay")
with c4:
    label_d = st.text_input("Parameter D", "Organic matter")

labels = [
    label_a.strip() or "A",
    label_b.strip() or "B",
    label_c.strip() or "C",
    label_d.strip() or "D"
]

if len(set(labels)) != 4:
    st.error("The four parameter names must be different.")
    st.stop()


# ============================================================
# 2. Subgroup definition
# ============================================================

st.header("2. Define subgroup fields")

definition_mode = st.radio(
    "Subgroup definition mode",
    ["Manual input", "Upload Excel file"],
    horizontal=True
)

subgroup_defs = []

if definition_mode == "Manual input":

    if "subgroup_count_v3" not in st.session_state:
        st.session_state.subgroup_count_v3 = 1

    b1, b2, _ = st.columns([1, 1, 4])

    with b1:
        if st.button("+ Add subgroup"):
            st.session_state.subgroup_count_v3 += 1
            st.rerun()

    with b2:
        if st.button("− Remove last subgroup") and st.session_state.subgroup_count_v3 > 1:
            st.session_state.subgroup_count_v3 -= 1
            st.rerun()

    for i in range(st.session_state.subgroup_count_v3):
        with st.expander(f"Subgroup {i+1}", expanded=True):
            name = st.text_input(
                "Subgroup name",
                f"Subgroup {i+1}",
                key=f"name_v3_{i}"
            )

            ranges = []
            for j, label in enumerate(labels):
                left, right = st.columns(2)
                with left:
                    lo = st.number_input(
                        f"{label} minimum",
                        min_value=0,
                        max_value=100,
                        value=0,
                        step=1,
                        key=f"lo_v3_{i}_{j}"
                    )
                with right:
                    hi = st.number_input(
                        f"{label} maximum",
                        min_value=0,
                        max_value=100,
                        value=100,
                        step=1,
                        key=f"hi_v3_{i}_{j}"
                    )
                ranges.append((int(lo), int(hi)))

            sum_min = sum(r[0] for r in ranges)
            sum_max = sum(r[1] for r in ranges)

            if sum_min > 100 or sum_max < 100:
                st.error(
                    f"No 100% composition is possible: sum of minima = {sum_min}%, "
                    f"sum of maxima = {sum_max}%."
                )
            else:
                st.success(
                    f"Range check OK: minima sum to {sum_min}% and maxima to {sum_max}%."
                )

            subgroup_defs.append({
                "name": name.strip() or f"Subgroup {i+1}",
                "ranges": ranges
            })

else:
    st.markdown("""
Upload an Excel file containing one subgroup per row.

**Required columns:**

`Subgroup | A_min | A_max | B_min | B_max | C_min | C_max | D_min | D_max`

Example:

| Subgroup | A_min | A_max | B_min | B_max | C_min | C_max | D_min | D_max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Sandy | 55 | 75 | 10 | 30 | 5 | 20 | 2 | 10 |
| Loamy | 30 | 50 | 25 | 45 | 15 | 30 | 3 | 10 |
| Clayey | 15 | 35 | 20 | 40 | 35 | 55 | 3 | 10 |

The letters A–D refer to the parameter names defined above.
""")

    subgroup_file = st.file_uploader(
        "Upload subgroup definition file (.xlsx)",
        type=["xlsx"],
        key="subgroup_definition_uploader"
    )

    if subgroup_file is not None:
        try:
            sg_df = pd.read_excel(subgroup_file)

            # Clean Excel column names (spaces / non-breaking spaces / capitalization)
            sg_df.columns = (
                sg_df.columns
                .astype(str)
                .str.replace("\u00a0", " ", regex=False)
                .str.strip()
            )

            # Accept subgroup header case-insensitively
            sg_df = sg_df.rename(columns={
                c: "Subgroup"
                for c in sg_df.columns
                if c.strip().lower() == "subgroup"
            })

            required_cols = [
                "Subgroup",
                "A_min", "A_max",
                "B_min", "B_max",
                "C_min", "C_max",
                "D_min", "D_max"
            ]

            missing_cols = [c for c in required_cols if c not in sg_df.columns]

            if missing_cols:
                st.error("Missing required columns: " + ", ".join(missing_cols))
            else:
                sg_df = sg_df[required_cols].copy()
                numeric_cols = required_cols[1:]

                for col in numeric_cols:
                    sg_df[col] = pd.to_numeric(sg_df[col], errors="coerce")

                if sg_df[numeric_cols].isna().any().any():
                    st.error("At least one range value is missing or not numeric.")
                else:
                    invalid_rows = []

                    for idx, row in sg_df.iterrows():
                        name = str(row["Subgroup"]).strip()
                        if not name or name.lower() == "nan":
                            name = f"Subgroup {idx + 1}"

                        ranges = [
                            (int(row["A_min"]), int(row["A_max"])),
                            (int(row["B_min"]), int(row["B_max"])),
                            (int(row["C_min"]), int(row["C_max"])),
                            (int(row["D_min"]), int(row["D_max"]))
                        ]

                        range_error = any(
                            lo < 0 or hi > 100 or lo > hi
                            for lo, hi in ranges
                        )
                        sum_min = sum(r[0] for r in ranges)
                        sum_max = sum(r[1] for r in ranges)
                        closure_error = sum_min > 100 or sum_max < 100

                        if range_error or closure_error:
                            invalid_rows.append(
                                (idx + 2, name, sum_min, sum_max, range_error)
                            )

                        subgroup_defs.append({
                            "name": name,
                            "ranges": ranges
                        })

                    preview = sg_df.copy()
                    preview.columns = [
                        "Subgroup",
                        f"{labels[0]} min", f"{labels[0]} max",
                        f"{labels[1]} min", f"{labels[1]} max",
                        f"{labels[2]} min", f"{labels[2]} max",
                        f"{labels[3]} min", f"{labels[3]} max"
                    ]

                    st.success(f"{len(subgroup_defs)} subgroup(s) loaded.")
                    st.dataframe(preview, use_container_width=True)

                    for row_no, name, sum_min, sum_max, range_error in invalid_rows:
                        if range_error:
                            st.error(
                                f"Row {row_no} ({name}): range values must satisfy "
                                "0 ≤ minimum ≤ maximum ≤ 100."
                            )
                        if sum_min > 100 or sum_max < 100:
                            st.error(
                                f"Row {row_no} ({name}): no 100% composition is possible "
                                f"(sum minima = {sum_min}%, sum maxima = {sum_max}%)."
                            )

        except Exception as exc:
            st.error(f"Could not read subgroup definition file: {exc}")

# ============================================================
# 3. Generate subgroup fields automatically
# ============================================================

st.header("3. Generate subgroup fields")

# Always regenerate from the currently visible/manual or uploaded range definitions.
# This avoids stale Streamlit session-state data after switching input mode or
# updating the app code.
generated_subgroups = []

for sg in subgroup_defs:
    points = build_subgroup_points(sg["ranges"])
    generated_subgroups.append({
        "name": sg["name"],
        "ranges": sg["ranges"],
        "points": points
    })

if generated_subgroups:
    summary = pd.DataFrame([
        {
            "Subgroup": sg["name"],
            "Valid integer compositions": len(sg["points"]),
            "AB slices": sg["points"]["AB"].nunique() if not sg["points"].empty else 0
        }
        for sg in generated_subgroups
    ])

    st.dataframe(summary, use_container_width=True)

    empty = [sg["name"] for sg in generated_subgroups if sg["points"].empty]

    if empty:
        st.warning(
            "No valid compositions summing to 100 were generated for: "
            + ", ".join(empty)
        )
else:
    st.info("Define at least one valid subgroup field above.")

# ============================================================
# 4. Upload
# ============================================================

st.header("4. Upload dataset")

st.markdown(
    f"""
Excel columns can be either:

**A, B, C, D, Locality**

or:

**{labels[0]}, {labels[1]}, {labels[2]}, {labels[3]}, Locality**
"""
)

uploaded_file = st.file_uploader("Upload Excel file (.xlsx)", type=["xlsx"])


# ============================================================
# 5. Plot settings
# ============================================================

st.header("5. Plot settings")

pc1, pc2 = st.columns(2)
with pc1:
    point_size = st.slider("Sample point size", 5, 40, 18, 1)
with pc2:
    colorscale = st.selectbox(
        "Sample color scale",
        ["Plasma", "Viridis", "Turbo", "Inferno", "Cividis", "RdYlBu"]
    )

show_subgroups = st.checkbox("Show subgroup fields", value=True)
show_gray_grid = st.checkbox("Show gray Cantor grid", value=True)


# ============================================================
# 6. Plot
# ============================================================

if uploaded_file is not None:
    try:
        df = read_uploaded_dataset(uploaded_file, labels)
    except Exception as exc:
        st.error(str(exc))
        st.stop()

    df["Subgroup"] = df.apply(lambda r: classify_by_ranges(r, subgroup_defs), axis=1)

    fig = go.Figure()

    # Background first
    if show_gray_grid:
        add_gray_cantor_grid(fig)

    # Reference grid lines
    for y in range(10, 100, 10):
        fig.add_shape(
            type="line",
            x0=0,
            x1=RECTANGLES[-1][0] + RECTANGLES[-1][1] + 10,
            y0=y,
            y1=y,
            line=dict(color="rgba(80,80,80,0.45)", width=0.8, dash="dash"),
            layer="above"
        )

    for x in [442, 909.5, 1352, 1769.5, 2162, 2529.5, 2872, 3189.5, 3482,
              3749.5, 3992, 4209.5, 4402, 4569.5, 4712, 4850.5, 4922, 4995.5, 5037.5]:
        fig.add_shape(
            type="line",
            x0=x, x1=x, y0=0, y1=100,
            line=dict(color="rgba(80,80,80,0.55)", width=0.8, dash="dash"),
            layer="above"
        )

    # User-defined subgroup fields over the gray grid
    if show_subgroups and generated_subgroups:
        nonempty_subgroups = [
            sg for sg in generated_subgroups if not sg["points"].empty
        ]
        add_subgroup_fields(fig, nonempty_subgroups)

        if nonempty_subgroups:
            st.caption(
                "Subgroup fields drawn: "
                + ", ".join(sg["name"] for sg in nonempty_subgroups)
            )

    # Uploaded samples
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
        for loc, a, b, c, d, sg in zip(
            df["Locality"], df["A"], df["B"], df["C"], df["D"], df["Subgroup"]
        )
    ]

    # Outer black ring
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=point_size + 3,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=2.5)
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            showlegend=False
        )
    )

    # Colored sample marker
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=point_size,
                color=ratio,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                opacity=0.95,
                colorbar=dict(
                    title=f"{labels[0]} / ({labels[0]} + {labels[1]})",
                    thickness=20
                )
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Uploaded samples"
        )
    )

    # Small black centre point, as in the garnet application
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=max(3.0, point_size * 0.22),
                color="black",
                line=dict(width=0)
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    tickvals = list(X_LABELS.keys())
    ticktext = [
        f"{ab}<br>CD{str(100 - int(ab[2:])).zfill(2)}"
        for ab in X_LABELS.values()
    ]

    fig.update_layout(
        plot_bgcolor="white",
        paper_bgcolor="white",
        height=950,
        xaxis=dict(
            title=f"Sum of {labels[0]} (%) + {labels[1]} (%)",
            range=[-30, RECTANGLES[-1][0] + RECTANGLES[-1][1] + 20],
            tickvals=tickvals,
            ticktext=ticktext,
            tickangle=0,
            tickfont=dict(size=12, color="black"),
            title_font=dict(size=22, color="black"),
            showgrid=False,
            zeroline=False,
            linecolor="black",
            linewidth=2
        ),
        yaxis=dict(
            title=f"{labels[2]} (%) /// {labels[3]} (%) = grid height − {labels[2]} (%)",
            range=[0, 100],
            dtick=10,
            tickfont=dict(size=12, color="black"),
            title_font=dict(size=20, color="black"),
            showgrid=False,
            zeroline=False,
            linecolor="black",
            linewidth=2
        ),
        legend=dict(
            title="<b>Subgroups</b>",
            x=0.02,
            y=0.98,
            bgcolor="rgba(255,255,255,0.88)",
            bordercolor="black",
            borderwidth=1
        ),
        margin=dict(l=90, r=120, t=40, b=100),
        hoverlabel=dict(font_size=16)
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Normalized uploaded data")
    display_df = df[["Locality", "A", "B", "C", "D", "Subgroup"]].rename(
        columns={
            "A": labels[0],
            "B": labels[1],
            "C": labels[2],
            "D": labels[3]
        }
    )
    st.dataframe(display_df, use_container_width=True)

else:
    st.info("Upload an Excel file to display samples. The subgroup fields can be generated beforehand.")
