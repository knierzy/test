import itertools
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st

st.set_page_config(layout="wide", page_title="Cantor Grids")

st.title("Cantor Grids – Four-Parameter Compositional Visualization")
st.caption("Build: V36 — Aitchison / Log-Euclidean subgroup distance choice")
st.caption(
    "Define four compositional parameters, create subgroup fields from parameter ranges, "
    "and optionally add sample points manually or from Excel."
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
}

SUBGROUP_COLORS = [
    "#4E79A7", "#F28E2B", "#E15759", "#76B7B2", "#59A14F",
    "#EDC948", "#B07AA1", "#FF9DA7", "#9C755F", "#BAB0AC",
    "#1F77B4", "#FF7F0E", "#2CA02C", "#D62728", "#9467BD",
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


def rgba_with_alpha(color, alpha):
    """
    Return an rgba(...) string with the requested alpha.

    Supports both:
    - rgba(r,g,b,a)
    - rgb(r,g,b)
    - hexadecimal colors such as #4E79A7
    """
    color = str(color).strip()

    if color.startswith("#"):
        hex_color = color.lstrip("#")
        if len(hex_color) == 3:
            hex_color = "".join(ch * 2 for ch in hex_color)
        if len(hex_color) != 6:
            raise ValueError(f"Unsupported hex color: {color}")

        r = int(hex_color[0:2], 16)
        g = int(hex_color[2:4], 16)
        b = int(hex_color[4:6], 16)
        return f"rgba({r},{g},{b},{alpha})"

    if color.startswith("rgba("):
        parts = color.replace("rgba(", "").replace(")", "").split(",")
        return f"rgba({parts[0].strip()},{parts[1].strip()},{parts[2].strip()},{alpha})"

    if color.startswith("rgb("):
        parts = color.replace("rgb(", "").replace(")", "").split(",")
        return f"rgba({parts[0].strip()},{parts[1].strip()},{parts[2].strip()},{alpha})"

    raise ValueError(f"Unsupported color format: {color}")


def add_subgroup_fields(fig, subgroup_results, hull_width=1.0, subfield_width=1.0, color_map=None):
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

        color = (color_map or {}).get(sg["name"], SUBGROUP_COLORS[idx % len(SUBGROUP_COLORS)])

        # Softer garnet-style subgroup appearance:
        # subtle fill, moderately transparent subfield outline,
        # and an even softer dashed outer hull.
        fill = rgba_with_alpha(color, 0.12)
        subfield_line_color = rgba_with_alpha(color, 0.70)
        hull_line_color = rgba_with_alpha(color, 0.60)

        first_trace = True

        for ab, group in pts.groupby("AB"):
            x_min = float(group["x"].min()) - 0.45
            x_max = float(group["x"].max()) + 0.45
            y_min = float(group["y"].min()) - 0.45
            y_max = float(group["y"].max()) + 0.45

            # Clip subgroup rectangle to the valid Cantor-grid slice
            row = int(99 - ab)
            row_start, height, _ = RECTANGLES[row]

            valid_x_min = float(row_start)
            valid_x_max = float(row_start + height)
            valid_y_min = 0.0
            valid_y_max = float(100 - ab)

            x_min = max(x_min, valid_x_min)
            x_max = min(x_max, valid_x_max)
            y_min = max(y_min, valid_y_min)
            y_max = min(y_max, valid_y_max)

            # Colored rectangular subfield, matching the garnet-style display.
            fig.add_trace(
                go.Scatter(
                    x=[x_min, x_min, x_max, x_max, x_min],
                    y=[y_min, y_max, y_max, y_min, y_min],
                    mode="lines",
                    line=dict(color=subfield_line_color, width=subfield_width),
                    fill="toself",
                    fillcolor=fill,
                    name=sg["name"],
                    legendgroup=sg["name"],
                    showlegend=False,
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

        # Thin outer boundary around the complete subgroup:
        # convex hull of all generated/scattered subfield positions.
        hull = convex_hull_2d(pts[["x", "y"]].to_numpy())
        if len(hull) >= 3:
            hull_x = [p[0] for p in hull] + [hull[0][0]]
            hull_y = [p[1] for p in hull] + [hull[0][1]]
            fig.add_trace(
                go.Scatter(
                    x=hull_x,
                    y=hull_y,
                    mode="lines",
                    line=dict(color=hull_line_color, width=hull_width, dash="dash"),
                    fill=None,
                    hoverinfo="skip",
                    legendgroup=sg["name"],
                    showlegend=False
                )
            )
        elif len(hull) == 2:
            fig.add_trace(
                go.Scatter(
                    x=[hull[0][0], hull[1][0]],
                    y=[hull[0][1], hull[1][1]],
                    mode="lines",
                    line=dict(color=hull_line_color, width=hull_width, dash="dash"),
                    hoverinfo="skip",
                    legendgroup=sg["name"],
                    showlegend=False
                )
            )


def subgroup_rectangles_by_ab(sg):
    """
    Return the visible rectangular subgroup field for each AB slice.

    The geometry is calculated exactly like in add_subgroup_fields(), including
    the +/- 0.45 padding and clipping to the valid Cantor-grid slice.
    """
    pts = sg["points"]
    rectangles = {}

    if pts.empty:
        return rectangles

    for ab, group in pts.groupby("AB"):
        x_min = float(group["x"].min()) - 0.45
        x_max = float(group["x"].max()) + 0.45
        y_min = float(group["y"].min()) - 0.45
        y_max = float(group["y"].max()) + 0.45

        row = int(99 - ab)
        row_start, height, _ = RECTANGLES[row]

        valid_x_min = float(row_start)
        valid_x_max = float(row_start + height)
        valid_y_min = 0.0
        valid_y_max = float(100 - ab)

        x_min = max(x_min, valid_x_min)
        x_max = min(x_max, valid_x_max)
        y_min = max(y_min, valid_y_min)
        y_max = min(y_max, valid_y_max)

        if x_max > x_min and y_max > y_min:
            rectangles[int(ab)] = (x_min, x_max, y_min, y_max)

    return rectangles


def calculate_subgroup_field_overlaps(subgroup_results):
    """
    Calculate actual geometric overlap of the visible subgroup fields.

    Pairwise and three-way intersections are evaluated. For every overlapping
    combination, the shared intersection area is expressed separately as a
    percentage of EACH participating subgroup field:

        overlap of A = intersection(A,B,...) / area(A) * 100

    Thus asymmetric overlap is preserved. Example:
        A–B: A 30% | B 60%

    For a three-way overlap:
        A–B–C: A 20% | B 35% | C 55%
    """
    valid = [sg for sg in subgroup_results if not sg["points"].empty]

    rect_maps = {
        sg["name"]: subgroup_rectangles_by_ab(sg)
        for sg in valid
    }

    total_areas = {}
    for sg in valid:
        name = sg["name"]
        total_areas[name] = sum(
            (x_max - x_min) * (y_max - y_min)
            for x_min, x_max, y_min, y_max in rect_maps[name].values()
        )

    overlaps = []

    # Evaluate pairwise and three-way overlaps.
    max_order = min(3, len(valid))

    for order in range(2, max_order + 1):
        for combo in itertools.combinations(valid, order):
            names = [sg["name"] for sg in combo]
            maps = [rect_maps[name] for name in names]

            common_abs = set(maps[0].keys())
            for rect_map in maps[1:]:
                common_abs &= set(rect_map.keys())

            intersection_area = 0.0

            for ab in common_abs:
                rects = [rect_map[ab] for rect_map in maps]

                overlap_x = max(
                    0.0,
                    min(r[1] for r in rects) - max(r[0] for r in rects)
                )
                overlap_y = max(
                    0.0,
                    min(r[3] for r in rects) - max(r[2] for r in rects)
                )

                intersection_area += overlap_x * overlap_y

            if intersection_area <= 0:
                continue

            percentages = {
                name: (
                    intersection_area / total_areas[name] * 100.0
                    if total_areas[name] > 0 else 0.0
                )
                for name in names
            }

            overlaps.append({
                "Names": names,
                "Order": order,
                "Intersection_area": intersection_area,
                "Percentages": percentages,
                # Sorting score: strongest affected participating subgroup.
                "Max_percent": max(percentages.values()),
            })

    return sorted(
        overlaps,
        key=lambda row: (row["Order"], row["Max_percent"]),
        reverse=True
    )


def add_overlap_hatching(
    fig,
    subgroup_results,
    hatch_spacing=0.65,
    hatch_alpha=0.70,
    hatch_width=1.60,
    outline_alpha=0.75,
    outline_width=2.20
):
    """
    Highlight actual pairwise overlap areas with a clearly visible but still
    restrained red diagonal hatch plus a thin dashed red overlap boundary.

    Performance-optimized:
    all hatch segments are collected into one Scatter trace and all overlap
    outlines into a second Scatter trace, instead of creating many shapes.
    """
    valid = [sg for sg in subgroup_results if not sg["points"].empty]

    rect_maps = {
        sg["name"]: subgroup_rectangles_by_ab(sg)
        for sg in valid
    }

    drawn_regions = set()
    hatch_x = []
    hatch_y = []
    outline_x = []
    outline_y = []

    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            name_a = valid[i]["name"]
            name_b = valid[j]["name"]

            rects_a = rect_maps[name_a]
            rects_b = rect_maps[name_b]
            common_abs = set(rects_a).intersection(rects_b)

            for ab in common_abs:
                ax0, ax1, ay0, ay1 = rects_a[ab]
                bx0, bx1, by0, by1 = rects_b[ab]

                x0 = max(ax0, bx0)
                x1 = min(ax1, bx1)
                y0 = max(ay0, by0)
                y1 = min(ay1, by1)

                if x1 <= x0 or y1 <= y0:
                    continue

                region_key = (
                    int(ab),
                    round(x0, 4), round(x1, 4),
                    round(y0, 4), round(y1, 4)
                )
                if region_key in drawn_regions:
                    continue
                drawn_regions.add(region_key)

                # Thin dashed outline around the true overlap rectangle.
                outline_x.extend([x0, x0, x1, x1, x0, None])
                outline_y.extend([y0, y1, y1, y0, y0, None])

                # Diagonal hatch lines with positive slope.
                k_min = y0 - x1
                k_max = y1 - x0
                k = k_min

                while k <= k_max + 1e-9:
                    pts = []

                    y_at_x0 = x0 + k
                    if y0 <= y_at_x0 <= y1:
                        pts.append((x0, y_at_x0))

                    y_at_x1 = x1 + k
                    if y0 <= y_at_x1 <= y1:
                        pts.append((x1, y_at_x1))

                    x_at_y0 = y0 - k
                    if x0 <= x_at_y0 <= x1:
                        pts.append((x_at_y0, y0))

                    x_at_y1 = y1 - k
                    if x0 <= x_at_y1 <= x1:
                        pts.append((x_at_y1, y1))

                    unique = []
                    for p in pts:
                        if not any(
                            abs(p[0] - q[0]) < 1e-9 and
                            abs(p[1] - q[1]) < 1e-9
                            for q in unique
                        ):
                            unique.append(p)

                    if len(unique) >= 2:
                        best_pair = None
                        best_d2 = -1.0

                        for a in range(len(unique)):
                            for b in range(a + 1, len(unique)):
                                dx = unique[a][0] - unique[b][0]
                                dy = unique[a][1] - unique[b][1]
                                d2 = dx * dx + dy * dy

                                if d2 > best_d2:
                                    best_d2 = d2
                                    best_pair = (unique[a], unique[b])

                        if best_pair is not None:
                            (sx, sy), (ex, ey) = best_pair
                            hatch_x.extend([sx, ex, None])
                            hatch_y.extend([sy, ey, None])

                    k += hatch_spacing

    if hatch_x:
        fig.add_trace(
            go.Scatter(
                x=hatch_x,
                y=hatch_y,
                mode="lines",
                line=dict(
                    color=f"rgba(210,35,35,{hatch_alpha})",
                    width=hatch_width
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Subgroup overlap hatching"
            )
        )

    if outline_x:
        fig.add_trace(
            go.Scatter(
                x=outline_x,
                y=outline_y,
                mode="lines",
                line=dict(
                    color=f"rgba(190,20,20,{outline_alpha})",
                    width=outline_width,
                    dash="dot"
                ),
                hoverinfo="skip",
                showlegend=False,
                name="Subgroup overlap boundary"
            )
        )


def dynamic_axis_font_size(text, base_size, min_size):
    """
    Scale an axis-title font according to the visible title length.
    The title always remains on a single line; only the font size changes.
    """
    n = len(str(text))

    if n <= 30:
        return base_size
    elif n <= 45:
        return max(min_size, int(base_size * 0.88))
    elif n <= 60:
        return max(min_size, int(base_size * 0.75))
    elif n <= 80:
        return max(min_size, int(base_size * 0.62))
    elif n <= 100:
        return max(min_size, int(base_size * 0.52))
    else:
        return min_size


def build_dynamic_axis_titles(labels):
    """
    Build x/y titles from the current parameter names.

    Both titles remain on a single line. Long parameter names are handled
    automatically by reducing the corresponding axis-title font size.
    """
    x_title = f"Sum of {labels[0]} (%) + {labels[1]} (%)"
    y_title = (
        f"{labels[2]} (%) /// {labels[3]} (%) = "
        f"grid height − {labels[2]} (%)"
    )

    x_size = dynamic_axis_font_size(
        x_title,
        base_size=35,
        min_size=14
    )
    y_size = dynamic_axis_font_size(
        y_title,
        base_size=28,
        min_size=14
    )

    return x_title, y_title, x_size, y_size


def subgroup_statistics_from_generated(subgroup_results):
    """
    Calculate mean compositions and standard deviations for each subgroup
    from all valid integer compositions generated from its parameter ranges.

    These statistics define the diagonal-covariance Mahalanobis approximation.
    """
    means = {}
    sigmas = {}

    for sg in subgroup_results:
        pts = sg["points"]

        if pts.empty:
            continue

        comp = pts[["A", "B", "C", "D"]].astype(float)

        means[sg["name"]] = comp.mean(axis=0).to_numpy()

        # ddof=1 when possible; replace undefined/zero values below.
        if len(comp) > 1:
            sigma = comp.std(axis=0, ddof=1).to_numpy()
        else:
            sigma = np.zeros(4, dtype=float)

        # Same practical safeguard as in the garnet application:
        # avoid singular/infinite weighting for extremely narrow ranges.
        sigma = np.where(np.isfinite(sigma), sigma, 0.0)
        sigma = np.clip(sigma, 0.5, None)

        sigmas[sg["name"]] = sigma

    return means, sigmas


def log_euclidean_distance(mu_i, mu_j):
    """
    Log-Euclidean distance between two four-component subgroup centroids.

    Each component is transformed with ln(x + 1), so zeros are allowed:
        d = sqrt(sum((ln(mu_i + 1) - ln(mu_j + 1))^2))
    """
    x = np.asarray(mu_i, dtype=float)
    y = np.asarray(mu_j, dtype=float)

    return float(
        np.sqrt(
            np.sum(
                (np.log1p(x) - np.log1p(y)) ** 2
            )
        )
    )


def multiplicative_zero_replacement(comp, delta=0.5):
    """
    Replace zero components in a closed percentage composition multiplicatively.

    Each zero receives delta percentage points. The original positive components
    are reduced proportionally so that the total remains 100% and the ratios
    among the originally positive components remain unchanged.
    """
    comp = np.asarray(comp, dtype=float)

    if np.any(~np.isfinite(comp)) or np.any(comp < 0) or comp.sum() <= 0:
        return None

    comp = comp / comp.sum() * 100.0
    zero_mask = comp <= 0
    n_zero = int(zero_mask.sum())

    if n_zero == 0:
        return comp

    delta = float(delta)
    if not np.isfinite(delta) or delta <= 0:
        return None

    replacement_total = n_zero * delta
    if replacement_total >= 100.0:
        return None

    positive_mask = ~zero_mask
    positive_total = float(comp[positive_mask].sum())
    if positive_total <= 0:
        return None

    result = comp.copy()
    result[zero_mask] = delta

    scale = (100.0 - replacement_total) / positive_total
    result[positive_mask] = comp[positive_mask] * scale

    return result


def aitchison_distance(mu_i, mu_j, zero_replacement=0.5):
    """
    Aitchison distance between two four-component subgroup centroids.

    Zeros are handled by multiplicative zero replacement before the centered
    log-ratio (CLR) transformation. The replacement value is expressed in
    percentage points and defaults to 0.5%.
    """
    x = multiplicative_zero_replacement(mu_i, delta=zero_replacement)
    y = multiplicative_zero_replacement(mu_j, delta=zero_replacement)

    if x is None or y is None:
        return np.nan

    def clr(comp):
        logs = np.log(comp)
        return logs - logs.mean()

    clr_x = clr(x)
    clr_y = clr(y)
    return float(np.linalg.norm(clr_x - clr_y))


def subgroup_reference_distance_colors(
    subgroup_results,
    colorscale,
    reference_name=None,
    distance_metric="Aitchison",
    aitchison_zero_replacement=0.5
):
    """
    For plots without sample points:
    1) calculate the mean A/B/C/D composition of every subgroup,
    2) calculate all pairwise distances using the selected metric,
    3) use either a user-selected reference subgroup or, by default,
       automatically choose the subgroup with the largest mean distance
       to all other subgroups,
    4) color every subgroup continuously by its distance from that reference.

    Supported metrics:
    - Aitchison (default; CLR geometry with 0.5% zero replacement)
    - Log-Euclidean (ln(x + 1))
    """
    means, sigmas = subgroup_statistics_from_generated(subgroup_results)
    names = [sg["name"] for sg in subgroup_results if sg["name"] in means]

    if not names:
        return None, {}, {}, means, sigmas

    if len(names) == 1:
        ref_name = names[0]
        distances = {ref_name: 0.0}
        color_map = {ref_name: sample_colorscale(colorscale, [0.0])[0]}
        return ref_name, distances, color_map, means, sigmas

    pairwise = {name: {} for name in names}

    for i, name_i in enumerate(names):
        for j, name_j in enumerate(names):
            if i == j:
                pairwise[name_i][name_j] = 0.0
            elif name_j not in pairwise[name_i]:
                if distance_metric == "Aitchison":
                    d = aitchison_distance(
                        means[name_i],
                        means[name_j],
                        zero_replacement=aitchison_zero_replacement
                    )
                else:
                    d = log_euclidean_distance(
                        means[name_i],
                        means[name_j]
                    )
                pairwise[name_i][name_j] = d
                pairwise[name_j][name_i] = d

    mean_distance = {
        name: float(
            np.mean(
                [d for other, d in pairwise[name].items() if other != name]
            )
        )
        for name in names
    }

    # Default: most compositionally distinct subgroup.
    automatic_ref_name = max(mean_distance, key=mean_distance.get)

    # User selection overrides the automatic reference when valid.
    if reference_name in names:
        ref_name = reference_name
    else:
        ref_name = automatic_ref_name

    distances = {
        name: float(pairwise[ref_name][name])
        for name in names
    }

    max_distance = max(distances.values()) if distances else 0.0

    if max_distance > 0:
        normalized = {
            name: d / max_distance
            for name, d in distances.items()
        }
    else:
        normalized = {name: 0.0 for name in names}

    color_map = {
        name: sample_colorscale(
            colorscale,
            [normalized[name]]
        )[0]
        for name in names
    }

    return ref_name, distances, color_map, means, sigmas


def classify_diagonal_mahalanobis(df_input, subgroup_results):
    """
    Assign every uploaded composition to the closest subgroup using
    Mahalanobis distance with a diagonal covariance approximation:

        d = sqrt(sum(((x_i - mu_i) / sigma_i)^2))

    Means and sigmas are calculated from each subgroup's constrained
    Cartesian product (A+B+C+D=100).
    """
    means, sigmas = subgroup_statistics_from_generated(subgroup_results)

    labels_out = []
    distances_out = []

    if not means:
        df_input["Nearest_Subfield"] = "Unclassified"
        df_input["Mahalanobis_Distance"] = np.nan
        return df_input, means, sigmas

    X = df_input[["A", "B", "C", "D"]].astype(float).to_numpy()

    for x in X:
        best_label = None
        best_distance = np.inf

        for name, mu in means.items():
            sigma = sigmas[name]
            d = float(np.sqrt(np.sum(((x - mu) / sigma) ** 2)))

            if d < best_distance:
                best_distance = d
                best_label = name

        labels_out.append(best_label)
        distances_out.append(best_distance)

    df_input["Nearest_Subfield"] = labels_out
    df_input["Mahalanobis_Distance"] = distances_out

    return df_input, means, sigmas


def classification_summary(df_input, subgroup_results):
    """
    Return count and percentage for every defined subgroup.
    """
    subgroup_names = [
        sg["name"] for sg in subgroup_results if not sg["points"].empty
    ]

    counts = df_input["Nearest_Subfield"].value_counts()
    n = len(df_input)

    rows = []
    for name in subgroup_names:
        count = int(counts.get(name, 0))
        pct = (count / n * 100.0) if n else 0.0
        rows.append({
            "Subgroup": name,
            "Points": count,
            "Percent": round(pct, 1)
        })

    return pd.DataFrame(rows)


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
    label_a = st.text_input("Parameter A", "A")
with c2:
    label_b = st.text_input("Parameter B", "B")
with c3:
    label_c = st.text_input("Parameter C", "C")
with c4:
    label_d = st.text_input("Parameter D", "D")

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
| Subgroup 1 | 55 | 75 | 10 | 30 | 5 | 20 | 2 | 10 |
| Subgroup 2 | 30 | 50 | 25 | 45 | 15 | 30 | 3 | 10 |
| Subgroup 3 | 15 | 35 | 20 | 40 | 35 | 55 | 3 | 10 |

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
                    st.dataframe(preview, use_container_width=False)

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

    st.dataframe(summary, use_container_width=False)

    empty = [sg["name"] for sg in generated_subgroups if sg["points"].empty]

    if empty:
        st.warning(
            "No valid compositions summing to 100 were generated for: "
            + ", ".join(empty)
        )
else:
    st.info("Define at least one valid subgroup field above.")

# ============================================================
# 4. Optional sample points
# ============================================================

st.header("4. Optional sample points")

sample_mode = st.radio(
    "Sample point input",
    ["No sample points", "Manual input", "Upload Excel file"],
    horizontal=True,
    help=(
        "Sample points are optional. Choose 'No sample points' to display only "
        "the Cantor grid and subgroup fields."
    )
)

df = None

if sample_mode == "Manual input":
    st.caption(
        "Enter one four-parameter composition. Values are automatically normalized "
        "to 100% using the Largest Remainder Method."
    )

    m1, m2, m3, m4, m5 = st.columns([1, 1, 1, 1, 2])

    with m1:
        manual_a = st.number_input(
            labels[0], min_value=0.0, value=25.0, step=1.0, key="manual_point_a"
        )
    with m2:
        manual_b = st.number_input(
            labels[1], min_value=0.0, value=25.0, step=1.0, key="manual_point_b"
        )
    with m3:
        manual_c = st.number_input(
            labels[2], min_value=0.0, value=25.0, step=1.0, key="manual_point_c"
        )
    with m4:
        manual_d = st.number_input(
            labels[3], min_value=0.0, value=25.0, step=1.0, key="manual_point_d"
        )
    with m5:
        manual_locality = st.text_input(
            "Locality / sample name", "Manual sample", key="manual_point_locality"
        )

    try:
        manual_norm = normalize_to_100_lrm(
            [manual_a, manual_b, manual_c, manual_d]
        )
        mx, my = calculate_final_position(*manual_norm)

        df = pd.DataFrame([{
            "A": int(manual_norm[0]),
            "B": int(manual_norm[1]),
            "C": int(manual_norm[2]),
            "D": int(manual_norm[3]),
            "Locality": manual_locality.strip() or "Manual sample",
            "x": mx,
            "y": my,
        }])

        st.caption(
            "Normalized composition: "
            f"{labels[0]}={manual_norm[0]}%, "
            f"{labels[1]}={manual_norm[1]}%, "
            f"{labels[2]}={manual_norm[2]}%, "
            f"{labels[3]}={manual_norm[3]}%."
        )
    except Exception as exc:
        st.error(str(exc))
        df = None

elif sample_mode == "Upload Excel file":
    st.markdown(
        f"""
Excel columns can be either:

**A, B, C, D, Locality**

or:

**{labels[0]}, {labels[1]}, {labels[2]}, {labels[3]}, Locality**
"""
    )

    uploaded_file = st.file_uploader(
        "Upload sample dataset (.xlsx)",
        type=["xlsx"],
        key="sample_dataset_uploader"
    )

    if uploaded_file is not None:
        try:
            df = read_uploaded_dataset(uploaded_file, labels)
        except Exception as exc:
            st.error(str(exc))
            df = None

# ============================================================
# 5. Plot settings
# ============================================================

st.header("5. Plot settings")

pc1, pc2, pc3, pc4 = st.columns(4)
with pc1:
    point_size = st.slider("Sample point size", 3, 40, 7, 1)
with pc2:
    colorscale = st.selectbox(
        "Sample color scale",
        ["Plasma", "Viridis", "Turbo", "Inferno", "Cividis", "RdYlBu"]
    )
with pc3:
    subgroup_hull_width = st.slider(
        "Subgroup convex hull line width",
        min_value=0.2,
        max_value=5.0,
        value=0.8,
        step=0.1,
        help="Controls the thickness of the dashed outer convex-hull line around each subgroup."
    )
with pc4:
    subgroup_subfield_width = st.slider(
        "Subfield boundary line width",
        min_value=0.2,
        max_value=5.0,
        value=1.0,
        step=0.1,
        help="Controls the thickness of the colored boundary line around each individual scattered subfield."
    )

legend_scale = st.slider(
    "Statistics box size factor",
    min_value=0.5,
    max_value=1.8,
    value=0.8,
    step=0.05,
    help="Scales the in-plot statistics box and text. Useful when many subgroups are defined."
)

show_subgroups = st.checkbox("Show subgroup fields", value=True)
show_subgroup_labels = st.checkbox(
    "Show subgroup labels (first two letters)",
    value=False,
    help="Places the first two letters of each subgroup name at the center of its generated field."
)
show_gray_grid = st.checkbox("Show gray Cantor grid", value=True)
show_overlap_hatching = st.checkbox(
    "Highlight subgroup overlap",
    value=True,
    help="Adds a subtle diagonal hatch only where subgroup fields geometrically overlap."
)

# Distance metric for subgroup-to-subgroup comparison.
# Aitchison is the default because A/B/C/D are compositional percentages.
distance_metric = st.selectbox(
    "Subgroup distance metric",
    ["Aitchison", "Log-Euclidean"],
    index=0,
    help=(
        "Aitchison is recommended for compositional percentage data and uses CLR "
        "geometry. Zero components are handled by multiplicative replacement. "
        "Log-Euclidean uses ln(x + 1)."
    )
)

if distance_metric == "Aitchison":
    aitchison_zero_replacement = st.number_input(
        "Aitchison zero replacement δ (%)",
        min_value=0.01,
        max_value=10.0,
        value=0.5,
        step=0.05,
        format="%.2f",
        help=(
            "Each exact zero is replaced by δ percentage points. The positive "
            "components are reduced proportionally so that the composition remains "
            "closed and their mutual ratios are preserved. For integer percentage "
            "data, 0.5% is a practical default."
        )
    )
else:
    aitchison_zero_replacement = 0.5

distance_title = (
    "Aitchison distance"
    if distance_metric == "Aitchison"
    else "Log-Euclidean distance"
)

# Reference subgroup for the selected distance metric.
available_reference_groups = [
    sg["name"] for sg in generated_subgroups if not sg["points"].empty
]

reference_mode_options = ["Automatic (largest mean distance)"] + available_reference_groups

selected_reference_option = st.selectbox(
    "Reference subgroup",
    reference_mode_options,
    index=0,
    help=(
        f"Automatic selects the subgroup with the largest mean {distance_title.lower()} "
        "to all other subgroups. Alternatively, choose any subgroup as the reference "
        "for the displayed distances and color scale."
    )
)

selected_reference_name = (
    None
    if selected_reference_option == "Automatic (largest mean distance)"
    else selected_reference_option
)


# ============================================================
# 6. Plot
# ============================================================

# Sample-based classification is only calculated when sample points are present.
has_samples = df is not None and not df.empty

if has_samples:
    reference_subgroup = None
    subgroup_reference_distances = {}
    subgroup_distance_color_map = {}

    df, subgroup_means, subgroup_sigmas = classify_diagonal_mahalanobis(
        df,
        generated_subgroups
    )

    df["Inside_Range_Field"] = df.apply(
        lambda r: classify_by_ranges(r, subgroup_defs),
        axis=1
    )

    df["Subgroup"] = df["Nearest_Subfield"]
    summary_df = classification_summary(df, generated_subgroups)

    if "Locality" in df.columns and len(df) > 0:
        first_locality = str(df["Locality"].iloc[0])
    else:
        first_locality = "not specified"
else:
    (
        reference_subgroup,
        subgroup_reference_distances,
        subgroup_distance_color_map,
        subgroup_means,
        subgroup_sigmas,
    ) = subgroup_reference_distance_colors(
        generated_subgroups,
        colorscale,
        reference_name=selected_reference_name,
        distance_metric=distance_metric,
        aitchison_zero_replacement=aitchison_zero_replacement
    )

    summary_df = pd.DataFrame(columns=["Subgroup", "Points", "Percent"])
    first_locality = "no sample points"

reference_is_automatic = selected_reference_name is None


PLOT_WIDTH = 1700
PLOT_HEIGHT = 950

# Pairwise and three-way geometric overlap of visible subgroup fields.
# Percentages are reported separately relative to each participating field.
subgroup_field_overlaps = calculate_subgroup_field_overlaps(generated_subgroups)

# Keep the in-plot text box compact when many subgroup pairs overlap.
MAX_OVERLAP_LINES = 6
displayed_overlaps = subgroup_field_overlaps[:MAX_OVERLAP_LINES]
hidden_overlap_count = max(0, len(subgroup_field_overlaps) - len(displayed_overlaps))

if has_samples:
    # Dynamic statistics-box sizing.
    # Height grows with subgroup count, while the user can scale the whole box.
    n_subgroups = max(len(summary_df), 1)
    title_fs = int(40 * legend_scale)
    method_fs = int(23 * legend_scale)
    locality_fs = int(30 * legend_scale)
    square_fs = int(42 * legend_scale)
    group_fs = int(31 * legend_scale)
    count_fs = int(26 * legend_scale)

    # Dynamic vertical statistics-box size.
    # Calculate the required height in PIXELS from the actual font sizes and
    # convert it to Plotly paper coordinates. This keeps the rectangle around
    # the complete legend even when the plot height or legend scale changes.

    title_line_px = title_fs * 1.35
    method_line_px = method_fs * 1.45
    locality_line_px = locality_fs * 1.35

    # Each subgroup row must accommodate the large coloured square as well as
    # the group/count text. A little extra spacing prevents glyphs touching.
    subgroup_line_px = max(
        square_fs * 1.10,
        group_fs * 1.40,
        count_fs * 1.40
    )

    # The HTML contains <br><br> after the method block and after locality.
    gap_after_methods_px = 20 * legend_scale
    gap_before_groups_px = 26 * legend_scale

    # Top/bottom breathing room inside the white rectangle.
    padding_top_px = 18 * legend_scale
    padding_bottom_px = 24 * legend_scale

    overlap_header_px = (24 * legend_scale) if displayed_overlaps else 0
    overlap_line_px = (22 * legend_scale) if displayed_overlaps else 0
    overlap_extra_lines = len(displayed_overlaps) + (1 if hidden_overlap_count > 0 else 0)
    overlap_block_px = (
        (18 * legend_scale)
        + overlap_header_px
        + overlap_extra_lines * overlap_line_px
        if displayed_overlaps
        else 0
    )

    legend_height_px = (
        padding_top_px
        + title_line_px
        + 2 * method_line_px
        + gap_after_methods_px
        + locality_line_px
        + gap_before_groups_px
        + n_subgroups * subgroup_line_px
        + overlap_block_px
        + padding_bottom_px
    )

    legend_height = legend_height_px / PLOT_HEIGHT

    # Keep the rectangle inside the plotting domain.
    legend_height = min(0.94, max(0.24, legend_height))

    legend_y1 = 0.98
    legend_y0 = max(0.02, legend_y1 - legend_height)

    # Approximate width from longest rendered legend entry.
    # Width follows the longest complete visible entry, not separate name/count maxima.
    complete_entries = []
    for _, _row in summary_df.iterrows():
        complete_entries.append(
            f"{_row['Subgroup']} {int(_row['Points'])} points ({float(_row['Percent']):.1f}%)"
        )

    overlap_entries = [
        " – ".join(o["Names"]) + ": " + " | ".join(
            f"{name} {o['Percentages'][name]:.1f}%"
            for name in o["Names"]
        )
        for o in displayed_overlaps
    ]

    longest_entry_chars = max(
        [len("Subgroup Classification"),
         len("Classification -> Mahalanobis distance"),
         len("(diagonal covariance approximation)"),
         len(f"Locality: {first_locality}"),
         len("Subgroup field overlap (shared area as % of each field)")]
        + [len(x) for x in complete_entries]
        + [len(x) for x in overlap_entries]
    )

    # Character-based width estimate with padding. Cap prevents the box from
    # swallowing the whole plot for unusually long labels.
    legend_width = min(
        0.82,
        max(0.36, (0.10 + longest_entry_chars * 0.0080) * legend_scale)
    )
    legend_x0 = 0.015
    legend_x1 = min(0.92, legend_x0 + legend_width)

    # Build in-plot statistical summary text exactly in the style of the garnet application
    stats_legend_text = (
        f"<span style='font-size:{title_fs}px; font-weight:bold;'>Subgroup Classification</span><br>"
        f"<span style='font-size:{method_fs}px; font-style:italic;'>Classification -> Mahalanobis distance</span><br>"
        f"<span style='font-size:{method_fs}px; font-style:italic;'>(diagonal covariance approximation)</span><br><br>"
        f"<span style='font-size:{locality_fs}px;'>Locality: {first_locality}</span><br><br>"
    )

    for idx, row in summary_df.iterrows():
        subgroup_name = row["Subgroup"]
        count = int(row["Points"])
        pct = float(row["Percent"])
        color = SUBGROUP_COLORS[idx % len(SUBGROUP_COLORS)]

        stats_legend_text += (
            f'<span style="color:{color}; font-size:{square_fs}px; vertical-align:middle;">■</span> '
            f'<span style="font-size:{group_fs}px; font-weight:bold; vertical-align:middle;">{subgroup_name}</span> '
            f'<span style="font-size:{count_fs}px; vertical-align:middle;">'
            f'{count} points ({pct:.1f}%)'
            f'</span><br>'
        )

    if displayed_overlaps:
        overlap_fs = int(20 * legend_scale)
        overlap_header_fs = int(22 * legend_scale)
        stats_legend_text += (
            f"<br><span style='font-size:{overlap_header_fs}px; font-weight:bold;'>"
            "Subgroup field overlap</span>"
            f"<span style='font-size:{overlap_fs}px; font-style:italic;'> "
            "(shared area as % of each field)</span><br>"
        )

        for overlap in displayed_overlaps:
            overlap_names = " – ".join(overlap["Names"])
            overlap_values = " | ".join(
                f"{name} {overlap['Percentages'][name]:.1f}%"
                for name in overlap["Names"]
            )
            stats_legend_text += (
                f"<span style='font-size:{overlap_fs}px;'>"
                f"{overlap_names}: {overlap_values}"
                "</span><br>"
            )

        if hidden_overlap_count > 0:
            stats_legend_text += (
                f"<span style='font-size:{overlap_fs}px; font-style:italic;'>"
                f"+ {hidden_overlap_count} additional overlapping pair(s)"
                "</span><br>"
            )

else:
    # Keep a subgroup legend box visible even when no sample points are plotted.
    n_subgroups = max(len([sg for sg in generated_subgroups if not sg["points"].empty]), 1)

    title_fs = int(40 * legend_scale)
    square_fs = int(42 * legend_scale)
    group_fs = int(31 * legend_scale)

    subgroup_line_px = max(square_fs * 1.10, group_fs * 1.40)
    padding_top_px = 18 * legend_scale
    padding_bottom_px = 24 * legend_scale
    gap_before_groups_px = 18 * legend_scale

    overlap_header_px = (24 * legend_scale) if displayed_overlaps else 0
    overlap_line_px = (22 * legend_scale) if displayed_overlaps else 0
    overlap_extra_lines = len(displayed_overlaps) + (1 if hidden_overlap_count > 0 else 0)
    overlap_block_px = (
        (18 * legend_scale)
        + overlap_header_px
        + overlap_extra_lines * overlap_line_px
        if displayed_overlaps
        else 0
    )

    legend_height_px = (
        padding_top_px
        + title_fs * 1.35
        + gap_before_groups_px
        + n_subgroups * subgroup_line_px
        + overlap_block_px
        + padding_bottom_px
    )

    legend_height = min(0.94, max(0.18, legend_height_px / PLOT_HEIGHT))
    legend_y1 = 0.98
    legend_y0 = max(0.02, legend_y1 - legend_height)

    nonempty_names = [
        sg["name"] for sg in generated_subgroups if not sg["points"].empty
    ]
    overlap_entries = [
        " – ".join(o["Names"]) + ": " + " | ".join(
            f"{name} {o['Percentages'][name]:.1f}%"
            for name in o["Names"]
        )
        for o in displayed_overlaps
    ]

    longest_entry_chars = max(
        [
            len(distance_title),
            len(
                f"Reference: {reference_subgroup} "
                + ("(automatic: largest mean distance)" if reference_is_automatic else "(user selected)")
            ),
            len("Subgroup field overlap (shared area as % of each field)")
        ]
        + [len(name) + 9 for name in nonempty_names]
        + [len(x) for x in overlap_entries]
    )

    legend_width = min(
        0.62,
        max(0.28, (0.10 + longest_entry_chars * 0.0080) * legend_scale)
    )
    legend_x0 = 0.015
    legend_x1 = min(0.92, legend_x0 + legend_width)

    ref_text = reference_subgroup if reference_subgroup is not None else "not available"
    reference_note = (
        "automatic: largest mean distance"
        if reference_is_automatic
        else "user selected"
    )
    stats_legend_text = (
        f"<span style='font-size:{title_fs}px; font-weight:bold;'>{distance_title}</span><br>"
        f"<span style='font-size:{int(22 * legend_scale)}px; font-style:italic;'>"
        f"Reference: {ref_text} ({reference_note})</span><br><br>"
    )

    sorted_legend_subgroups = sorted(
        [sg for sg in generated_subgroups if not sg["points"].empty],
        key=lambda sg: subgroup_reference_distances.get(sg["name"], np.inf)
    )

    for idx, sg in enumerate(sorted_legend_subgroups):
        color = subgroup_distance_color_map.get(
            sg["name"], SUBGROUP_COLORS[idx % len(SUBGROUP_COLORS)]
        )
        distance = subgroup_reference_distances.get(sg["name"], np.nan)
        distance_txt = f"{distance:.2f}" if np.isfinite(distance) else "n/a"

        stats_legend_text += (
            f'<span style="color:{color}; font-size:{square_fs}px; vertical-align:middle;">■</span> '
            f'<span style="font-size:{group_fs}px; font-weight:bold; vertical-align:middle;">'
            f'{sg["name"]}</span> '
            f'<span style="font-size:{int(23 * legend_scale)}px;">d={distance_txt}</span><br>'
        )

    if displayed_overlaps:
        overlap_fs = int(20 * legend_scale)
        overlap_header_fs = int(22 * legend_scale)
        stats_legend_text += (
            f"<br><span style='font-size:{overlap_header_fs}px; font-weight:bold;'>"
            "Subgroup field overlap</span>"
            f"<span style='font-size:{overlap_fs}px; font-style:italic;'> "
            "(shared area as % of each field)</span><br>"
        )

        for overlap in displayed_overlaps:
            overlap_names = " – ".join(overlap["Names"])
            overlap_values = " | ".join(
                f"{name} {overlap['Percentages'][name]:.1f}%"
                for name in overlap["Names"]
            )
            stats_legend_text += (
                f"<span style='font-size:{overlap_fs}px;'>"
                f"{overlap_names}: {overlap_values}"
                "</span><br>"
            )

        if hidden_overlap_count > 0:
            stats_legend_text += (
                f"<span style='font-size:{overlap_fs}px; font-style:italic;'>"
                f"+ {hidden_overlap_count} additional overlapping pair(s)"
                "</span><br>"
            )

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
    active_subgroup_color_map = (
        subgroup_distance_color_map if not has_samples else None
    )
    add_subgroup_fields(
        fig,
        nonempty_subgroups,
        hull_width=subgroup_hull_width,
        subfield_width=subgroup_subfield_width,
        color_map=active_subgroup_color_map
    )

    if show_subgroup_labels:
        for i, sg in enumerate(nonempty_subgroups):
            pts = sg["points"]
            if pts.empty:
                continue

            # Use the centroid of all valid generated compositions as label position.
            label_x = float(pts["x"].mean())
            label_y = float(pts["y"].mean())

            # First two alphabetic characters of the subgroup name, upper case.
            letters = "".join(ch for ch in str(sg["name"]) if ch.isalpha())
            short_label = (letters[:2] if len(letters) >= 2 else letters).upper()

            # Use a text trace instead of a layout annotation.
            # The statistics box later replaces layout.annotations, which
            # previously removed these subgroup labels.
            fig.add_trace(
                go.Scatter(
                    x=[label_x],
                    y=[label_y],
                    mode="text",
                    text=[f"<b>{short_label}</b>"],
                    textposition="middle center",
                    textfont=dict(
                        size=20,
                        color="black",
                        family="Arial Black"
                    ),
                    hoverinfo="skip",
                    showlegend=False,
                    legendgroup=sg["name"]
                )
            )

    if nonempty_subgroups:
        st.caption(
            "Subgroup fields drawn: "
            + ", ".join(sg["name"] for sg in nonempty_subgroups)
        )

    if show_overlap_hatching and nonempty_subgroups:
        add_overlap_hatching(
            fig,
            nonempty_subgroups,
            hatch_spacing=0.65,
            hatch_alpha=0.70,
            hatch_width=1.60,
            outline_alpha=0.75,
            outline_width=2.20
        )

# Continuous subgroup-distance colorbar when no sample points are plotted.
if (not has_samples) and reference_subgroup is not None and subgroup_reference_distances:
    max_ref_distance = max(subgroup_reference_distances.values())

    # Invisible marker solely to render the continuous colorbar.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker=dict(
                size=0.1,
                color=[0.0],
                colorscale=colorscale,
                cmin=0.0,
                cmax=max(max_ref_distance, 1e-9),
                showscale=True,
                colorbar=dict(
                    title="",
                    thickness=20,
                    len=0.92,
                    y=0.5,
                    yanchor="middle",
                    x=1.035,
                    xanchor="left",
                    tickfont=dict(size=12)
                )
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # Vertical title beside the subgroup-distance colorbar.
    # Using an annotation instead of colorbar.title gives precise control
    # over rotation and spacing, avoiding overlap with tick labels.
    fig.add_annotation(
        x=1.082,
        y=0.5,
        xref="paper",
        yref="paper",
        text=f"{distance_title} from {reference_subgroup}",
        textangle=-90,
        showarrow=False,
        font=dict(size=14, color="black"),
        xanchor="center",
        yanchor="middle",
        align="center"
    )

if has_samples:
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
            f"Nearest subgroup: {sg}<br>"
            f"Mahalanobis distance: {dist:.3f}<br>"
            f"Inside defined range field: {inside}"
        )
        for loc, a, b, c, d, sg, dist, inside in zip(
            df["Locality"],
            df["A"], df["B"], df["C"], df["D"],
            df["Subgroup"],
            df["Mahalanobis_Distance"],
            df["Inside_Range_Field"]
        )
    ]

    # ========================================================
    # Layered sample markers
    # Outer ring = continuous colorbar value A / (A+B)
    # Inner core = nearest subgroup from minimum Mahalanobis distance
    # ========================================================

    subgroup_color_map = {
        sg["name"]: SUBGROUP_COLORS[i % len(SUBGROUP_COLORS)]
        for i, sg in enumerate(generated_subgroups)
        if not sg["points"].empty
    }

    inner_group_colors = [
        subgroup_color_map.get(name, "rgba(120,120,120,0.95)")
        for name in df["Subgroup"]
    ]

    outer_black_size = point_size + 10
    color_ring_size = point_size + 8

    # Larger inner subgroup core; no white separator ring.
    # This makes the categorical subgroup color visually dominant while
    # retaining a clearly visible continuous-value halo around it.
    inner_core_size = max(7, point_size + 3)
    center_dot_size = max(2.5, point_size * 0.22)

    # 1) Black outer frame
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=outer_black_size,
                color="rgba(0,0,0,0)",
                line=dict(color="black", width=1.5)
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # 2) Outer ring: continuous colorbar value
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=color_ring_size,
                color=ratio,
                colorscale=colorscale,
                cmin=0,
                cmax=1,
                opacity=0.95,
                line=dict(width=0),
                colorbar=dict(
                    title=f"{labels[0]} / ({labels[0]} + {labels[1]})",
                    thickness=20
                )
            ),
            hoverinfo="skip",
            showlegend=False
        )
    )

    # 3) Inner core: color of nearest subgroup by Mahalanobis distance
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=inner_core_size,
                color=inner_group_colors,
                line=dict(color="black", width=1),
                opacity=1
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
            name="Uploaded samples",
            showlegend=False
        )
    )

    # 4) Small black centre point
    fig.add_trace(
        go.Scatter(
            x=df["x"], y=df["y"],
            mode="markers",
            marker=dict(
                size=center_dot_size,
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

# Layout aligned more closely with the original garnet application.
# In particular, do not draw a heavy bottom x-axis line; instead use
# explicit top and left frame lines, as in the garnet plot.

x_axis_title, y_axis_title, x_axis_title_size, y_axis_title_size = build_dynamic_axis_titles(labels)

fig.update_layout(
    plot_bgcolor="white",
    paper_bgcolor="white",
    autosize=False,
    width=PLOT_WIDTH,
    height=PLOT_HEIGHT,
    xaxis=dict(
        title=dict(
            text=x_axis_title,
            font=dict(size=x_axis_title_size, color="black", family="Arial Black")
        ),
        range=[-30, RECTANGLES[-1][0] + RECTANGLES[-1][1] + 20],
        tickvals=tickvals,
        ticktext=ticktext,
        tickangle=0,
        tickfont=dict(size=16, color="black"),
        automargin=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks=""
    ),
    yaxis=dict(
        title=dict(
            text=y_axis_title,
            font=dict(size=y_axis_title_size, color="black", family="Arial Black")
        ),
        range=[0, 100],
        constrain="domain",
        dtick=10,
        tickfont=dict(size=16, color="black"),
        automargin=True,
        showgrid=False,
        zeroline=False,
        showline=False,
        ticks=""
    ),
    showlegend=False,
    margin=dict(l=0, r=70, t=20, b=5),
    hoverlabel=dict(font_size=24)
)

# Garnet-style plot frame: left vertical border and thin top border.
x_min_frame = -30
x_max_frame = RECTANGLES[-1][0] + RECTANGLES[-1][1]

fig.add_shape(
    type="line",
    x0=x_min_frame,
    x1=x_min_frame,
    y0=0,
    y1=100,
    line=dict(color="black", width=3),
    layer="above"
)

fig.add_shape(
    type="line",
    x0=x_min_frame,
    x1=x_max_frame,
    y0=100,
    y1=100,
    line=dict(color="black", width=2),
    layer="above"
)

# ========================================================
# IN-PLOT STATISTICS / SUBGROUP BOX
# ========================================================

# White rectangle behind the statistics text
fig.add_shape(
    type="rect",
    xref="paper",
    yref="paper",
    x0=legend_x0,
    x1=legend_x1,
    y0=legend_y0,
    y1=legend_y1,
    fillcolor="white",
    line=dict(color="black", width=3),
    layer="above"
)

# IMPORTANT: use update_layout(annotations=[...]) exactly as in the garnet script
fig.update_layout(
    annotations=[
        dict(
            x=legend_x0,
            y=legend_y1,
            xref="paper",
            yref="paper",
            text=stats_legend_text,
            showarrow=False,
            font=dict(size=max(16, int(28 * legend_scale)), color="black"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            xanchor="left",
            yanchor="top",
            align="left",
            textangle=0
        )
    ],
    showlegend=False
)



st.plotly_chart(fig, use_container_width=True)

# ========================================================
# Export figure — same dimensions as the displayed figure
# ========================================================

st.subheader("Export figure")

export_format = st.selectbox(
    "Export format",
    ["PNG", "SVG"],
    key="cantor_export_format"
)

try:
    img_bytes = fig.to_image(
        format=export_format.lower(),
        width=PLOT_WIDTH,
        height=PLOT_HEIGHT,
        scale=2
    )

    if export_format == "PNG":
        file_extension = "png"
        mime_type = "image/png"
    else:
        file_extension = "svg"
        mime_type = "image/svg+xml"

    st.download_button(
        label=f"Download {export_format}",
        data=img_bytes,
        file_name=f"cantor_grid.{file_extension}",
        mime=mime_type
    )

except Exception as exc:
    st.warning(
        "Figure export is currently unavailable. "
        "For PNG/SVG export, make sure Kaleido is installed. "
        f"Details: {exc}"
    )

if has_samples:
    # ========================================================
    # Statistical classification output
    # ========================================================
    st.subheader("Distance-based subgroup classification")

    st.caption(
        "Classification -> Mahalanobis distance using a diagonal "
        "covariance approximation. Subgroup means and standard deviations "
        "are calculated from the valid integer compositions generated from "
        "the specified subgroup ranges."
    )

    if not summary_df.empty:
        st.dataframe(summary_df, use_container_width=False)

    if subgroup_means:
        stats_rows = []
        for sg_name in subgroup_means:
            mu = subgroup_means[sg_name]
            sigma = subgroup_sigmas[sg_name]

            stats_rows.append({
                "Subgroup": sg_name,
                f"{labels[0]} mean": round(float(mu[0]), 2),
                f"{labels[0]} SD": round(float(sigma[0]), 2),
                f"{labels[1]} mean": round(float(mu[1]), 2),
                f"{labels[1]} SD": round(float(sigma[1]), 2),
                f"{labels[2]} mean": round(float(mu[2]), 2),
                f"{labels[2]} SD": round(float(sigma[2]), 2),
                f"{labels[3]} mean": round(float(mu[3]), 2),
                f"{labels[3]} SD": round(float(sigma[3]), 2),
            })

        with st.expander("Show subgroup means and standard deviations"):
            st.dataframe(pd.DataFrame(stats_rows), use_container_width=True)

    st.subheader("Normalized uploaded data")
    display_df = df[
        [
            "Locality", "A", "B", "C", "D",
            "Subgroup", "Mahalanobis_Distance", "Inside_Range_Field"
        ]
    ].rename(
        columns={
            "A": labels[0],
            "B": labels[1],
            "C": labels[2],
            "D": labels[3]
        }
    )
    st.dataframe(display_df, use_container_width=False)

else:
    st.caption(
        "No sample points selected. The plot shows the Cantor grid and subgroup fields only."
    )
