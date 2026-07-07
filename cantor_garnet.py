# This script constructs a Cantor diagram for the garnet endmember system
# almandine–spessartine–pyrope–grossular. Scattered provenance fields
# (±1 standard deviation ranges), derived primarily from Suggate and Hall (2014),
# are displayed. The script allows plotting garnet compositions
# onto the diagram and assigns each data point to the most likely provenance
# field. Provenance assignment is computed statistically via Mahalanobis distance using subfield means and standard
# deviations.


import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np

st.set_page_config(layout="wide")

st.title("Cantor Diagram – Garnet Provenance Classification")

st.markdown("""
**Required column order:**  
1. Almandine   
2. Spessartine   
3. Pyrope   
4. Grossular  
5. Locality

**A template Excel file (garnet_examples_streamlit.xlsx) is available online at the following link. It shows the required column order and data format.  
https://github.com/knierzy/cantor_grids/blob/main/data/garnet_examples_streamlit.xlsx

If you use this application in scientific work, please cite:

Knierzinger, W., Forstner, S. J., Eder, A. (2026). Cantor grids: A new 2D visualization framework for four-parameter compositional datasets in geochemistry and soil science. 
Journal of Geochemical Exploration 289, 108129. https://doi.org/10.1016/j.gexplo.2026.108129
""")

uploaded_file = st.file_uploader(
    "Upload Excel file (.xlsx)",
    type=["xlsx"]
)

if uploaded_file is None:
    st.stop()

convex_hulls_file_4 = "data/convex_hull_amphibolites_general.xlsx"
convex_hulls_file_5 = "data/convex_hull_greenschists_.xlsx"
convex_hulls_file_6 = "data/convex_hull_granites_.xlsx"
convex_hulls_file_7 = "data/convex_hull_blueschists_.xlsx"
convex_hulls_file_8 = "data/convex_hull_calc_silicate_rocks_.xlsx"
convex_hulls_file_9 = "data/convex_hull_granulites_general.xlsx"
convex_hulls_file_11 = "data/convex_hull_eclogites_.xlsx"
convex_hulls_file_12 = "data/convex_hull_ultramafic_.xlsx"

# color mappings
color_mapping_files = {
    convex_hulls_file_5: "rgba(144, 238, 144, 0.9)",  # Greenschists - light green
    convex_hulls_file_4: "rgba(75, 50, 35, 0.9)",  # Amphibolites - black
    convex_hulls_file_11: "rgba(0, 100, 0, 0.9)",  # Eclogites - dark green
    convex_hulls_file_7: "rgba(0, 0, 255, 0.9)",  # Blueschists - blue
    convex_hulls_file_8: "rgba(64, 224, 208, 0.9)",  # Calc-Silicate Rocks - turquoise
    convex_hulls_file_6: "rgba(255, 215, 0, 0.9)",  # Granites - yellow
    convex_hulls_file_9: "rgba(255, 140, 0, 0.9)",  # Granulites General - orange
    convex_hulls_file_12: "rgba(205, 92, 92, 0.9)"  # Ultramafic - brick red
}

# legend mapping
legend_mapping = {
    convex_hulls_file_6: "Granites & Pegmatites",
    convex_hulls_file_5: "Greenschists",
    convex_hulls_file_4: "Amphibolites",
    convex_hulls_file_7: "Blueschists",
    convex_hulls_file_8: "Calc-silicate rocks",
    convex_hulls_file_9: "Granulites",
    convex_hulls_file_11: "Eclogites",
    convex_hulls_file_12: "Ultramafic rocks"
}

# Rectangle data with the classification system up to AB1
rechtecke = [
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

# Set up diagram
fig = go.Figure()

# normalization LRM
def normalize_to_100_LRM(row):
    cols = ['Alm', 'Prp', 'Grs', 'Sps']
    values = row[cols].astype(float).to_numpy()

    ints = np.floor(values).astype(int)
    remainders = values - ints
    missing = 100 - ints.sum()

    if missing > 0:
        order = np.argsort(-remainders)
        for i in range(missing):
            ints[order[i]] += 1

    elif missing < 0:
        order = np.argsort(remainders)
        for i in range(-missing):
            ints[order[i]] -= 1

    row[cols] = ints
    return row


def ensure_transparency(color, alpha=0.7):
    if "rgba" in color:
        return color[:color.rfind(",")] + f", {alpha})"
    elif color.startswith("#"):
        r = int(color[1:3], 16)
        g = int(color[3:5], 16)
        b = int(color[5:7], 16)
        return f"rgba({r}, {g}, {b}, {alpha})"
    else:
        return f"rgba(0, 0, 0, {alpha})"


def add_rechtecke_mit_farbverlauf(rechtecke, x_offset, spiegeln=False):
    for i, (y_position, hoehe, label) in enumerate(rechtecke):
        breite = i + 1
        gradient_steps = 5 if i >= 90 else 10

        grau_start = 200
        grau_ende = 230

        for step in range(gradient_steps):
            grau_wert = int(grau_start + (grau_ende - grau_start) * (step / (gradient_steps - 1)))
            alpha = 0.8 - (0.6 * (step / (gradient_steps - 1)))
            color = f'rgba({grau_wert}, {grau_wert}, {grau_wert}, {alpha})'

            y_start = y_position + (step / gradient_steps) * hoehe
            y_end = y_position + ((step + 1) / gradient_steps) * hoehe

            if spiegeln:
                x_start, x_end = x_offset - breite, x_offset
            else:
                x_start, x_end = x_offset, x_offset + breite

            fig.add_trace(go.Scatter(
                x=[x_start, x_start, x_end, x_end, x_start],
                y=[y_start, y_end, y_end, y_start, y_start],
                fill="toself",
                mode="none",
                fillcolor=color,
                hoverinfo="skip",
                showlegend=False
))

        for x_pos in range(1, breite):
            x_val = x_offset - x_pos if spiegeln else x_offset + x_pos
            fig.add_trace(go.Scatter(
                x=[x_val, x_val],
               y=[y_position, y_position + hoehe],
               mode="lines",
                line=dict(color="gray", width=2),
                 showlegend=False
           ))

# === Create axis labels 
x_labels = {
    50: "AB99", 440: "AB95", 915: "AB90", 1350: "AB85",
    1760: "AB80", 2158: "AB75", 2540: "AB70",
    2870: "AB65", 3195: "AB60", 3480: "AB55",
    3755: "AB50", 3995: "AB45", 4209: "AB40",
    4405: "AB35", 4570: "AB30", 4830: "AB20",
}

# === Update X-axis with labels ===
fig.update_layout(
    xaxis=dict(
        title="Summe A + B (%)",
        tickvals=list(x_labels.keys()),
        ticktext=[
            f"{ab}<br>CD{str(100 - int(ab[2:])).zfill(2)}"
            for ab in x_labels.values()
        ],
        tickangle=0,
    )
)

# Add rectangles
add_rechtecke_mit_farbverlauf(rechtecke, 0)


if uploaded_file is not None:

    # uploaded Excel file
    df_uploaded = pd.read_excel(uploaded_file)



if df_uploaded.shape[1] < 4:
    st.error("Excel file must contain at least 4 columns.")
    st.stop()

df_parameters = pd.DataFrame({

    "Alm": df_uploaded.iloc[:, 0],
    "Sps": df_uploaded.iloc[:, 1],
    "Prp": df_uploaded.iloc[:, 2],
    "Grs": df_uploaded.iloc[:, 3]

})

# optionale Locality-Spalte
if df_uploaded.shape[1] >= 5:
    df_parameters["Locality"] = df_uploaded.iloc[:, 4]

# Reihenfolge explizit fixieren
ordered_cols = ["Alm", "Sps", "Prp", "Grs"]

if "Locality" in df_parameters.columns:
    ordered_cols.append("Locality")

df_parameters = df_parameters[ordered_cols]

colorbar_choice = st.selectbox(
    "Select Color Scale",
    [
        "Plasma",
        "Viridis",
        "Turbo",
        "Inferno",
        "Cividis",
        "RdYlBu",
        "Custom"
    ]
)

distance_method = "Mahalanobis"


# POINT SIZE SLIDER


point_size = st.slider(
    "Select Point Size",
    min_value=5,
    max_value=40,
    value=20,
    step=1
)

outer_size = point_size
halo_size = point_size - 2
cutout_size = point_size * 0.08
inner_size = point_size * 0.08
center_size = point_size * 0.18


df_parameters = df_parameters[
    df_parameters.apply(
        lambda row: row[['Alm','Sps','Prp','Grs']].sum() >= 98,
        axis=1
    )
]


 # optional: second dataset empty
df_linz_params = pd.DataFrame(
    columns=[
        "Unnamed: 1",
        "Unnamed: 2",
        "Unnamed: 3",
        "Unnamed: 4"
    ]
)


if not df_linz_params.empty:
    df_linz_params = df_linz_params.astype(float).round().astype(int)
    df_linz_params = df_linz_params.apply(normalize_to_100_LRM, axis=1)

df_linz_params = df_linz_params.apply(normalize_to_100_LRM, axis=1)




df_parameters = df_parameters.apply(normalize_to_100_LRM, axis=1)



# Calculate AB (A + B) for the y-position
df_parameters['AB'] = df_parameters['Alm'] + df_parameters['Sps']


# Calculate the y-position based on AB and B
def calculate_y_position(ab_value, b_value):
    ab_index = 99 - int(ab_value)
    if ab_index >= 0 and ab_index < len(rechtecke):
        start_zeile = rechtecke[ab_index][0]
        hoehe = rechtecke[ab_index][1]
        y_position = start_zeile + b_value + 0.5
        return y_position
    return None

# Legend
legende_text = "<b>Garnet Provenance Groups</b><br>"


#  Manually ordered list of convex hulls in the legend
ordered_hulls = [
    convex_hulls_file_5,
    convex_hulls_file_4,
    convex_hulls_file_7,
    convex_hulls_file_9,
    convex_hulls_file_11,
    convex_hulls_file_6,
    convex_hulls_file_12,
    convex_hulls_file_8,
]

# Prepare legend text
legende_text += ""
for file_path in ordered_hulls:
    color = color_mapping_files[file_path]
    hull_name = legend_mapping.get(file_path, file_path.split("\\")[-1].split(".")[0])
    legende_text += f'<span style="color:{color};">■</span> {hull_name}<br>'


# List to group points by origin and rectangle (AB)
grouped_points = {}

# Load convex hull data
df_hulls_4 = pd.read_excel(convex_hulls_file_4)
df_hulls_5 = pd.read_excel(convex_hulls_file_5)
df_hulls_6 = pd.read_excel(convex_hulls_file_6)
df_hulls_7 = pd.read_excel(convex_hulls_file_7)
df_hulls_8 = pd.read_excel(convex_hulls_file_8)
df_hulls_9 = pd.read_excel(convex_hulls_file_9)
df_hulls_11 = pd.read_excel(convex_hulls_file_11)
df_hulls_12 = pd.read_excel(convex_hulls_file_12)

# Combine the files
df_hulls_combined = pd.concat([
    df_hulls_4, df_hulls_5, df_hulls_6, df_hulls_7,
    df_hulls_8, df_hulls_9, df_hulls_11, df_hulls_12
])

# Groups of hull data based on origin and AB value
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])


# Function to plot convex hulls with different colors based on source file
def plot_imported_hulls_with_file_colors(grouped_hulls, file_color_mapping):
    for (herkunft, ab_value), group in grouped_hulls:
        hull_x = group["X"].values
        hull_y = group["Y"].values

        hull_x = np.append(hull_x, hull_x[0])
        hull_y = np.append(hull_y, hull_y[0])

        # Determine the color based on the source file
        file_source = group["file_source"].iloc[0]
        color = file_color_mapping.get(file_source, "rgba(0, 0, 0, 0.5)")

        # Plot  convex hull
        fig.add_trace(go.Scatter(
            x=hull_x,
            y=hull_y,
            mode="lines",
            line=dict(color=color, width=1.5),
            fill="toself",
            fillcolor=ensure_transparency(color, alpha=0.4),
            name=f"Herkunft: {herkunft}, AB: {ab_value}"
        ))


def plot_linz_scatter(df):
    x_vals = []
    y_vals = []

    for _, row in df.iterrows():
        a = int(row['Unnamed: 1'])
        b = int(row['Unnamed: 2'])
        c = int(row['Unnamed: 3'])

        ab = a + b

        y_pos = calculate_y_position(ab, b)
        if y_pos is None:
            continue

        x_vals.append(c)
        y_vals.append(y_pos)

    fig.add_trace(go.Scatter(
        x=x_vals,
        y=y_vals,
        mode="markers",
        marker=dict(
            size=14,
            color="red",
            symbol="x",
            line=dict(color="black", width=1)
        ),
        name="Linz/Melk"
    ))

# Add a column to the DataFrames to mark the source file

df_hulls_4["file_source"] = convex_hulls_file_4
df_hulls_5["file_source"] = convex_hulls_file_5
df_hulls_6["file_source"] = convex_hulls_file_6
df_hulls_7["file_source"] = convex_hulls_file_7
df_hulls_8["file_source"] = convex_hulls_file_8
df_hulls_9["file_source"] = convex_hulls_file_9
df_hulls_11["file_source"] = convex_hulls_file_11
df_hulls_12["file_source"] = convex_hulls_file_12

# Combine the DataFrames
df_hulls_combined = pd.concat([df_hulls_4, df_hulls_5,df_hulls_6,df_hulls_7,df_hulls_8,df_hulls_9, df_hulls_11, df_hulls_12])

# Group the combined data
grouped_hulls_combined = df_hulls_combined.groupby(["Herkunft", "AB_Value"])

# Plot the imported convex hulls with file-specific colors

plot_imported_hulls_with_file_colors(grouped_hulls_combined, color_mapping_files)



if not df_linz_params.empty:
    df_linz_params['Ratio'] = (
        df_linz_params['Unnamed: 1']
        /
        (df_linz_params['Unnamed: 1'] + df_linz_params['Unnamed: 2'])
    )

# Calculate the ratio for color coding
df_parameters['Ratio'] = (
    df_parameters['Alm']
    /
    (df_parameters['Alm'] + df_parameters['Sps'])
)

custom_colorscale = [
    [0.0,  "#00007F"],   
    [0.20, "#007FFF"],  
    [0.40, "#00FFFF"],   
    [0.60, "#FFFF80"],   
    [0.80, "#FF9F40"],   
    [1.0,  "#FF4040"]    
]

if colorbar_choice == "Custom":
    selected_colorscale = custom_colorscale
else:
    selected_colorscale = colorbar_choice

# List to store values for the color legend
x_values, y_values, color_values = [], [], []
symbols = []   

# = Pernegg =
for idx, row in df_parameters.iterrows():
    a = row['Alm']
    b = row['Sps']
    c = row['Prp']
    ab_value = row['AB']
    ratio = row['Ratio']

    y_position_punkt = calculate_y_position(ab_value, b)
    if y_position_punkt is not None:
        x_values.append(c)
        y_values.append(y_position_punkt)
        color_values.append(ratio)
        symbols.append("circle")   

# = Linz/Melk =
for _, row in df_linz_params.iterrows():
    a = row['Unnamed: 1']
    b = row['Unnamed: 2']
    c = row['Unnamed: 3']
    ratio = row['Ratio']

    y_position_punkt = calculate_y_position(a + b, b)
    if y_position_punkt is not None:
        x_values.append(c)
        y_values.append(y_position_punkt)
        color_values.append(ratio)
        symbols.append("x")



x_vals = np.array(x_values)
y_vals = np.array(y_values)
ratios = np.array(color_values)
symbols_arr = np.array(symbols)

mask_circle = symbols_arr == "circle"
mask_cross = symbols_arr == "x"


#  CIRCLES (Pernegg)


# 1. Outer black ring
fig.add_trace(go.Scatter(
    x=x_vals[mask_circle],
    y=y_vals[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=outer_size,
        color="rgba(0,0,0,0)",
        line=dict(color="black", width=4)
    ),
    text=[
        f"Locality: {loc}<br>"
        f"Alm: {alm:.0f}%<br>"
        f"Sps: {sps:.0f}%<br>"
        f"Prp: {prp:.0f}%<br>"
        f"Grs: {grs:.0f}%<br>"
        for alm, sps, prp, grs, loc
        in zip(
            df_parameters["Alm"],
            df_parameters["Sps"],
            df_parameters["Prp"],
            df_parameters["Grs"],
            df_parameters["Locality"]
        )
    ],
    hovertemplate="%{text}<extra></extra>",
    showlegend=False
))

# 2.Colored Halo
fig.add_trace(go.Scatter(
    x=x_vals[mask_circle],
    y=y_vals[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=halo_size,
        color=ratios[mask_circle],
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        coloraxis="coloraxis",
        opacity=0.9,
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# 3. White Cutout
fig.add_trace(go.Scatter(
    x=x_vals[mask_circle],
    y=y_vals[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=cutout_size,
        color="white",
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# 4. Inner colored dot
fig.add_trace(go.Scatter(
    x=x_vals[mask_circle],
    y=y_vals[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=inner_size,
        color=ratios[mask_circle],
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        line=dict(color="black", width=1)
    ),
    showlegend=False
))

# 5.  central point
fig.add_trace(go.Scatter(
    x=x_vals[mask_circle],
    y=y_vals[mask_circle],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=center_size,
        color="black"
    ),
    hoverinfo="skip",
    showlegend=False
))


# X (Linz/Melk)


# 1. Black frame
fig.add_trace(go.Scatter(
    x=x_vals[mask_cross],
    y=y_vals[mask_cross],
    mode="markers",
    marker=dict(
        symbol="diamond",
        size=21,
        color=ratios[mask_cross],
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        coloraxis="coloraxis",
        line=dict(width=3)   
    ),
    showlegend=False
))

# 2. halo 
fig.add_trace(go.Scatter(
    x=x_vals[mask_cross],
    y=y_vals[mask_cross],
    mode="markers",
    marker=dict(
        symbol="diamond",
        size=18,
        color=ratios[mask_cross],
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        coloraxis="coloraxis",
        opacity=0.9,
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# 3.white Cutout
fig.add_trace(go.Scatter(
    x=x_vals[mask_cross],
    y=y_vals[mask_cross],
    mode="markers",
    marker=dict(
        symbol="diamond",
        size=1,
        color="white",
        line=dict(width=0)
    ),
    hoverinfo="skip",
    showlegend=False
))

# 4. Central point
fig.add_trace(go.Scatter(
    x=x_vals[mask_cross],
    y=y_vals[mask_cross],
    mode="markers",
    marker=dict(
        symbol="diamond",
        size=1,
        color=ratios[mask_cross],
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        line=dict(color="black", width=1)
    ),
    showlegend=False
))

# 5.  Central point
fig.add_trace(go.Scatter(
    x=x_vals[mask_cross],
    y=y_vals[mask_cross],
    mode="markers",
    marker=dict(
        symbol="circle",
        size=3.5,
        color="black"
    ),
    hoverinfo="skip",
    showlegend=False
))


#  COLORBAR (global!)


fig.update_layout(
    coloraxis=dict(
        colorscale=selected_colorscale,
        cmin=0,
        cmax=1,
        colorbar=dict(
            title="",
            thickness=20,
            len=0.9,
            y=0.5,
            yanchor="middle",
            tickfont=dict(size=24)
        )
    )
)


# Adjust layout
fig.update_layout(
plot_bgcolor="white",
    paper_bgcolor="white",
    xaxis=dict(
        title=dict(
            text="Sum of Almandine (%) + Spessartine (%)",
            font=dict(size=35, color="black", family="Arial Black")
        ),
        range=[0, rechtecke[-1][0] + rechtecke[-1][1]+ 20],
        tickformat=".0f",
        tickfont=dict(size=24, color="black"),
        showgrid=False,
        zeroline=False
    ),
    yaxis=dict(
        title=dict(
            text="Pyrope (%) /// Grossular (%) = height rectangle <sub>ABCD</sub> − Pyrope (%)",
            font=dict(size=28, color="black", family="Arial Black")
        ),
        range=[0, 100],
        constrain="domain",
        tickformat=".0f",
        dtick=10,
        tickfont=dict(size=24, color="black"),
        linecolor="gray",
        showgrid=False,
        zeroline=False
    ),
    autosize=False,
    width=2260,
    height=1210,
    margin=dict(l=0, r=5, t=20, b=5),
    showlegend=False
)

# Values for dashed lines
y_values = [ 10, 20, 30, 40, 50, 60, 70, 80, 90 ]

# Insert horizontal dashed lines
for y in y_values:
    fig.add_shape(
        type="line",
        x0=0, #
        x1=max(df_parameters['Grs']) + rechtecke[-1][0] + rechtecke[-1][1],  # Ending point of the line on the X-axis (right edge)
        y0=y,
        y1=y,
        line=dict(
            color="black",
            width=0.5,
            dash="dash"
        )
    )

# Positions for the vertical dashed lines
x_values = [909.5, 3749.5, 4569.5, 1769.5, 2529.5, 3189.5, 4209.5, 4850.5, 4995.5, 442, 1352, 2162, 2872, 3482, 3992, 4402, 4712, 4922, 5037.5]  # central positions of AB90, AB50, AB30

# Insert vertical dashed lines
for x in x_values:
    fig.add_shape(
        type="line",
        x0=x,
        x1=x,
        y0=0,
        y1=100,
        line=dict(
            color="black",
            width=1,
            dash="dash"
        )
    )

# Rotate the plot by 90 degrees (swap X and Y data)
for trace in fig.data:
    trace.x, trace.y = trace.y, trace.x



x_min = -30
x_max = rechtecke[-1][0] + rechtecke[-1][1]


fig.add_shape(
    type="line",
    x0=x_min,
    x1=x_min,
    y0=0,
    y1=100,
    line=dict(color="black", width=3)
)

def classify_dataset(df_input, means, sigmas):

    X_pts = df_input[["Alm","Sps","Prp","Grs"]].to_numpy()

    labels = []
    d_min  = []

    sigmas_safe = sigmas.clip(lower=0.5)
    invcovs = {
        k: np.diag(1.0/(sigmas_safe.loc[k].to_numpy()**2))
        for k in means.index
    }

    mus = {
        k: means.loc[k].to_numpy()
        for k in means.index
    }

    for x in X_pts:

        best_label = None
        best_d = np.inf

        for k in means.index:

            mu = mus[k]
            VI = invcovs[k]

            d2 = (x-mu) @ VI @ (x-mu).T

            if d2 < best_d:
                best_d = d2
                best_label = k

        labels.append(best_label)
        d_min.append(np.sqrt(best_d))

    df_input["Nearest_Subfield"] = labels
    df_input["Mahalanobis_Distance"] = d_min

    return df_input


def classify_dataset_logeuclidean(df_input, means):

    X_pts = df_input[["Alm","Sps","Prp","Grs"]].to_numpy()

    labels = []
    d_min = []

    mus = {
        k: means.loc[k].to_numpy()
        for k in means.index
    }

    for x in X_pts:

        best_label = None
        best_d = np.inf

        x_log = np.log(x + 1)

        for k in means.index:

            mu_log = np.log(mus[k] + 1)

            d = np.linalg.norm(x_log - mu_log)

            if d < best_d:
                best_d = d
                best_label = k

        labels.append(best_label)
        d_min.append(best_d)

    df_input["Nearest_Subfield_LogEuclidean"] = labels
    df_input["LogEuclidean_Distance"] = d_min

    return df_input

def clr_transform(x):

    x = np.asarray(x, dtype=float)

    eps = 1e-6
    x = x + eps

    gm = np.exp(np.mean(np.log(x)))

    return np.log(x / gm)


def classify_dataset_aitchison(df_input, means):

    X_pts = df_input[["Alm","Sps","Prp","Grs"]].to_numpy()

    labels = []
    d_min = []

    # CLR-transformed means
    mus_clr = {
        k: clr_transform(means.loc[k].to_numpy())
        for k in means.index
    }

    for x in X_pts:

        x_clr = clr_transform(x)

        best_label = None
        best_d = np.inf

        for k in means.index:

            mu_clr = mus_clr[k]

            d = np.linalg.norm(x_clr - mu_clr)

            if d < best_d:
                best_d = d
                best_label = k

        labels.append(best_label)
        d_min.append(best_d)

    df_input["Nearest_Subfield_Aitchison"] = labels
    df_input["Aitchison_Distance"] = d_min

    return df_input

from scipy.stats import chi2
import numpy as np

# SUBFIELD MEANS

subfield_means_raw = pd.DataFrame({
    "alm": [
        65.66982555,
        54.31709145,
        44.23111848,
        48.07424294,
        50.65111941,
        23.87808719,
        55.9267222,
        21.02162502
    ],

    "prp": [
        12.21280849,
        33.69541824,
        31.52675381,
        3.214631984,
        6.044688795,
        57.81881319,
        9.93239652,
        3.069200646
    ],

    "grs": [
        12.932808,
        10.07050389,
        23.08000919,
        22.67389225,
        2.962742713,
        17.48481604,
        24.99704974,
        64.92615389
    ],

    "sps": [
        9.18931335,
        1.921986915,
        1.177650051,
        27.99386213,
        40.3362223,
        0.812181278,
        9.156153933,
        10.98302044
    ]

}, index=[
    "Amphibolites",
    "Granulites",
    "Eclogites",
    "Greenschists",
    "Granites & Pegmatites",
    "Ultramafic rocks",
    "Blueschists",
    "Calc-silicate rocks"
])


# SUBFIELD STANDARD DEVIATIONS


subfield_sigmas_raw = pd.DataFrame({

    "alm": [
        10.34841545,
        12.29476235,
        12.67775939,
        17.88439765,
        14.35396254,
        8.963328014,
        11.23642444,
        21.2594703
    ],

    "prp": [
        8.402774083,
        14.64088125,
        12.21618211,
        2.160177633,
        5.404302193,
        15.99899131,
        7.608415516,
        2.57929375
    ],

    "grs": [
        9.198543982,
        8.220985049,
        8.401915528,
        9.018942259,
        1.638148843,
        13.22167668,
        5.837586628,
        30.103043
    ],

    "sps": [
        10.56094471,
        1.610341276,
        0.825854952,
        21.805533,
        17.31566399,
        0.575505324,
        13.55642173,
        14.93862155
    ]

}, index=subfield_means_raw.index)



means = (
    subfield_means_raw
    .rename(columns={
        "alm": "Alm",
        "prp": "Prp",
        "grs": "Grs",
        "sps":  "Sps"
    })
    [["Alm", "Sps", "Prp", "Grs"]]
)

sigmas = (
    subfield_sigmas_raw
    .rename(columns={
        "alm": "Alm",
        "prp": "Prp",
        "grs": "Grs",
        "sps":  "Sps"
    })
    [["Alm", "Sps", "Prp", "Grs"]]
)


# DIAGNOSTIC CHECK


print("\nCheck Spalten-Reihenfolge:")
print("Means cols:", list(means.columns))
print("Sigmas cols:", list(sigmas.columns))
print("Points cols: ['Alm','Pyr','Gro','Spe']")
# 



df_maha = df_parameters[["Alm", "Prp", "Grs", "Sps"]].copy()


if distance_method == "Mahalanobis":

    df_maha = classify_dataset(df_maha, means, sigmas)

    df_parameters["Nearest_Subfield"] = \
        df_maha["Nearest_Subfield"]

    df_parameters["Distance"] = \
        df_maha["Mahalanobis_Distance"]


elif distance_method == "Log-Euclidean":

    df_maha = classify_dataset_logeuclidean(
        df_maha,
        means
    )

    df_parameters["Nearest_Subfield"] = \
        df_maha["Nearest_Subfield_LogEuclidean"]

    df_parameters["Distance"] = \
        df_maha["LogEuclidean_Distance"]


elif distance_method == "Aitchison":

    df_maha = classify_dataset_aitchison(
        df_maha,
        means
    )

    df_parameters["Nearest_Subfield"] = \
        df_maha["Nearest_Subfield_Aitchison"]

    df_parameters["Distance"] = \
        df_maha["Aitchison_Distance"]
    

# Summary
summary_pf = df_parameters["Nearest_Subfield"].value_counts()
if not df_linz_params.empty:
    summary_lmf = df_linz_params["Nearest_Subfield"].value_counts()
    summary_lmf_pct = (summary_lmf / len(df_linz_params) * 100).round(1)
else:
    summary_lmf = pd.Series(dtype=float)
    summary_lmf_pct = pd.Series(dtype=float)

summary_pf_pct = (summary_pf / len(df_parameters) * 100).round(1)
summary_lmf_pct = (summary_lmf / len(df_linz_params) * 100).round(1)
print("\n=== Zusammenfassung (Mahalanobis, korrekt gemappt) ===")
print(pd.DataFrame({
    "PF Anzahl": summary_pf,
    "PF %": summary_pf_pct
}))

print(pd.DataFrame({
    "LMF Anzahl": summary_lmf,
    "LMF %": summary_lmf_pct
}))

#  small diagnostic output
print("\nCheck Spalten-Reihenfolge:")
print("Means cols:", list(means.columns))
print("Sigmas cols:", list(sigmas.columns))
print("Points cols: ['Alm','Sps','Prp','Grs']")

legend_text = (
    "<span style='font-size:40px; font-weight:bold;'>Garnet Provenance Groups</span><br>"
    "<span style='font-size:26px; font-style:italic;'>Classification based on standardized Mahalanobis distance using a diagonal covariance matrix</span><br><br>"
    f"<span style='font-size:30px;'>Locality: {df_parameters['Locality'].iloc[0] if 'Locality' in df_parameters.columns else 'not specified'}</span><br><br>"
)

for file_path in ordered_hulls:
    color = color_mapping_files[file_path]
    hull_name = legend_mapping[file_path]

    count_pf = summary_pf.get(hull_name, 0)
    pct_pf = summary_pf_pct.get(hull_name, 0)

    legend_text += (
        f'<span style="color:{color}; font-size:42px; vertical-align:middle;">■</span> '
        f'<span style="font-size:31px; font-weight:bold; vertical-align:middle;">{hull_name}</span> '
        f'<span style="font-size:26px; vertical-align:middle;">'
        f'{int(count_pf)} points ({pct_pf:.1f}%)'
        f'</span><br>'
    )

print("\nLegend content with counts and percentages:")
print(legend_text)

#Add both the colorbar title and the legend block 
fig.add_annotation(
    text="Almandine / (Almandine + Spessartine)",
    x=1.02,
    y=0.5,
    textangle=-90,
    showarrow=False,
    xref="paper",
    yref="paper",
    font=dict(size=30, color="black")
)

fig.add_shape(
    type="rect",
    xref="paper",
    yref="paper",
    x0=0.015,
    x1=0.37,
    y0=0.52,
    y1=0.98,
    fillcolor="white",
    line=dict(color="black", width=3),
    layer="above"
)


fig.update_layout(
    annotations=[
        dict(
            x=0.015,
            y=0.98,
            xref="paper",
            yref="paper",
            text=legend_text,
            showarrow=False,
            font=dict(size=28, color="black"),
            bgcolor="rgba(0,0,0,0)",
            borderwidth=0,
            xanchor="left",
            yanchor="top",
            align="left",
            textangle=0
        )
    ]
)

x_min = -30
x_max = rechtecke[-1][0] + rechtecke[-1][1]

fig.add_shape(
    type="line",
    x0=x_min,
    x1=x_max,
    y0=100,
    y1=100,
    line=dict(
        color="black",
        width=2
    )
)

fig.update_layout(
    hoverlabel=dict(
        font_size=24
    )
)

st.plotly_chart(fig, use_container_width=True)



fig.update_layout(
    margin=dict(
        l=140,   
        r=80,    
        t=20,    
        b=120    
    )
)


# IMAGE EXPORT: PNG or SVG


export_format = st.selectbox(
    "Select Export Format",
    ["PNG", "SVG"],
    key="export_format_selector"
)

st.write("Current export format:", export_format)

try:
    img_bytes = fig.to_image(
        format=export_format.lower(),
        width=2260,
        height=1210,
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
        file_name=f"cantor_diagram.{file_extension}",
        mime=mime_type,
        key=f"download_button_{file_extension}"
    )

except Exception as e:
    st.warning(
        f"{export_format} export not available on Streamlit Cloud "
        "(missing Chrome/Kaleido dependency)."
    )
