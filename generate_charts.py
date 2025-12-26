import os
import json
import random
import uuid
import numpy as np
from PIL import Image, ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUTPUT_DIR = "dataset_output_v1"
IMG_DIR = os.path.join(OUTPUT_DIR, "images")
os.makedirs(IMG_DIR, exist_ok=True)

class MetadataLibrary:
    X_AXIS_NAMES = [
        "Atmospheric Pressure (hPa)", "Concentration (mol/L)", "Elapsed Time (ms)", "Temperature (C)", "Wavelength (nm)",
        "Frequency (GHz)", "pH Level", "Voltage Input (V)", "Engine RPM", "Humidity (%)",
        "Current (A)", "Salinity (ppt)", "Depth (m)", "Angle (deg)", "Velocity (m/s)",
        "Particle Size (um)", "Oxygen Level (%)", "Radiation Dose (mSv)", "Force (N)", "Torque (Nm)",
        "Compression Ratio", "Mass Flow (kg/s)", "Light Intensity (lux)", "Viscosity (cP)", "Strain (%)",
        "Time Step", "Incubation Period (days)", "Signal Latency (ns)", "Load Factor", "Data Rate (Mbps)",
        "Sound Level (dB)", "Memory Usage (MB)", "Batch Size", "Learning Rate", "Wind Speed (knots)",
        "Carbon Content (%)", "Hardness (HRB)", "Damping Ratio", "Thermal Gradient", "Resistance (Ohm)",
        "Capacitance (uF)", "Magnetic Flux (mWb)", "Rotation Speed (rad/s)", "Pixel Pitch", "Buffer Size",
        "Altitude (km)", "Drill Speed", "Exposure Time (s)", "Coolant Flow", "Sample Index"
    ]
    Y_AXIS_NAMES = [
        "Efficiency (%)", "Throughput (Gbps)", "Response Time (us)", "Stability Index", "Heat Dissipation (W)",
        "Signal-to-Noise Ratio (dB)", "Refractive Index", "Growth Rate", "Energy Consumption (kWh)", "Precision Score",
        "Error Rate", "Tensile Strength (MPa)", "Yield (%)", "Conductivity (S/m)", "Amplitude",
        "Toxicity Level", "Absorption Ratio", "Conversion Gain", "Fuel Economy", "Packet Loss (%)",
        "Bit Error Rate (BER)", "F1 Score", "CPU Load (%)", "Density (g/cm3)", "Surface Tension",
        "Friction Coefficient", "Weight Loss (mg)", "Power Factor", "Luminous Efficacy", "Oxidation State",
        "Spectral Density", "Permeability", "Resonant Frequency", "Modulation Index", "Wait Time (s)",
        "Queue Length", "Service Life (h)", "Total Cost ($)", "Maintenance Priority", "Reliability (%)",
        "Compression Gain", "Burst Rate", "Phase Shift", "Clarity Index", "Sensitivity (V/K)",
        "Turbidity (NTU)", "Shear Stress (Pa)", "Porosity", "Reflectance", "Saturation"
    ]
    LEGEND_LABELS = [
        "Prototype Alpha", "Control Group", "Experimental Variant B", "Titanium Alloy", "Baseline Model",
        "Optimized Pipeline", "Legacy System", "Neural Network v2", "Hybrid Engine", "Composite A1",
        "Standard Unit", "Sensor Array X", "Reference Standard", "Algorithm V4", "Batch Alpha-9",
        "Polymer Matrix", "Graphene Layer", "Enhanced Cooling", "Default Setup", "Advanced Logic",
        "Candidate #102", "Service Node B", "Filtered Signal", "Raw Input", "Target Metric",
        "Model-K", "Solution-X", "Core Process", "Peripheral v3", "Master Control",
        "Active Cooling", "Passive Heat", "Primary Stage", "Backup Path", "Modified Core"
    ]


def get_infinite_line_pixel_coords(ax, fig, target_val, mode='x'):
    """
    返回 [x1, y1, x2, y2]，image-space 像素坐标
    与 create_bar_charts 完全一致的计算逻辑
    """
    fig.canvas.draw()
    # 这一步获取的高度必须是最终保存图片的高度
    img_h = fig.canvas.get_width_height()[1]

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if mode == 'x':  # 垂直线
        p_start = (target_val, ylim[0])
        p_end = (target_val, ylim[1])
    else:            # 水平线
        p_start = (xlim[0], target_val)
        p_end = (xlim[1], target_val)

    trans = ax.transData.transform([p_start, p_end])

    return [
        int(trans[0][0]),
        int(img_h - trans[0][1]),
        int(trans[1][0]),
        int(img_h - trans[1][1])
    ]


# =========================
# 绘制折线 / 散点图
# =========================
def create_precise_chart(run_id, x_data, curves, config, target_info):
    fig, ax = plt.subplots(figsize=(10, 6), dpi=100, layout='constrained')

    cmap = matplotlib.colormaps['tab20']
    colors = cmap(np.linspace(0, 1, len(curves)))
    chart_type = random.choice(['line', 'scatter', 'mixed'])
    line_styles = ['-', '--', '-.', ':']

    for i, (lbl, y_vals) in enumerate(curves.items()):
        if chart_type == 'scatter' or (chart_type == 'mixed' and i % 2 == 0):
            ax.scatter(x_data, y_vals, label=lbl, color=colors[i], s=30, alpha=0.7)
        else:
            ax.plot(
                x_data, y_vals,
                label=lbl,
                color=colors[i],
                linewidth=2,
                linestyle=random.choice(line_styles)
            )

    ax.set_xlabel(config['x_name'], fontweight='bold')
    ax.set_ylabel(config['y_name'], fontweight='bold')
    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend(loc='best', fontsize='small')

    # ✅ 像素坐标计算（此时 canvas 与最终图片一致）
    if target_info['mode'] == 'y_query':
        line_px = get_infinite_line_pixel_coords(
            ax, fig, target_info['y_val'], mode='y'
        )
    else:
        line_px = get_infinite_line_pixel_coords(
            ax, fig, target_info['x_val'], mode='x'
        )

    img_name = f"{run_id}.jpg"
    full_path = os.path.join(IMG_DIR, img_name)

    # ❗ 关键：不要 tight，移除 dpi 参数使用 figure 默认值
    plt.savefig(full_path)
    plt.close()

    return full_path, line_px



# =========================
# PIL 画红色参考线
# =========================
def draw_infinite_line(img_path, line_coords):
    img = Image.open(img_path).convert('RGB')
    draw = ImageDraw.Draw(img)

    draw.line(
        [(line_coords[0], line_coords[1]),
         (line_coords[2], line_coords[3])],
        fill=(255, 0, 0),
        width=3
    )

    traced_path = img_path.replace(".jpg", "_traced.jpg")
    img.save(traced_path)

    return [os.path.basename(img_path), os.path.basename(traced_path)]


# =========================
# QA 生成
# =========================
def generate_qa(x_data, curves, config):
    labels = list(curves.keys())
    idx = random.randint(10, 40)
    target_x = x_data[idx]

    qa_type = random.random()

    # 1. 数值查询
    if qa_type < 0.4:
        lbl = random.choice(labels)
        ans = curves[lbl][idx]
        question = f"When {config['x_name']} is {round(target_x, 2)}, what is the value of {lbl}?"
        target_info = {'mode': 'x_query', 'x_val': target_x}

    # 2. 差值查询
    elif qa_type < 0.7:
        l1, l2 = random.sample(labels, 2)
        ans = abs(curves[l1][idx] - curves[l2][idx])
        question = f"What is the absolute difference between {l1} and {l2} when {config['x_name']} is {round(target_x, 2)}?"
        target_info = {'mode': 'diff_query', 'x_val': target_x}

    # 3. 反向查询
    elif qa_type < 0.9:
        lbl = random.choice(labels)
        target_y = curves[lbl][idx]
        ans = target_x
        question = f"At what {config['x_name']} value does {lbl} reach approximately {round(target_y, 1)}?"
        target_info = {'mode': 'y_query', 'y_val': target_y}

    # 4. 比较
    else:
        l1, l2 = random.sample(labels, 2)
        ans = 1 if curves[l1][idx] > curves[l2][idx] else 0
        question = f"At {config['x_name']} = {round(target_x, 2)}, is {l1} higher than {l2}? (1=Yes, 0=No)"
        target_info = {'mode': 'comp_query', 'x_val': target_x}

    rounded_ans = round(ans, 1)
    options = {rounded_ans}
    while len(options) < 6:
        options.add(round(rounded_ans + random.choice([-10, -5, -2, -1, 1, 2, 5, 10]), 1))

    opts_list = sorted(list(options))
    return question, opts_list, rounded_ans, target_info


# =========================
# 主 Pipeline
# =========================
def run_pipeline(num_samples=20):
    lib = MetadataLibrary()
    results = []

    for i in range(num_samples):
        run_id = f"v7_{uuid.uuid4().hex[:6]}"

        config = {
            'x_name': random.choice(lib.X_AXIS_NAMES),
            'y_name': random.choice(lib.Y_AXIS_NAMES),
            'labels': random.sample(lib.LEGEND_LABELS, random.randint(2, 4))
        }

        x_data = np.linspace(random.randint(0, 10), random.randint(100, 200), 50)
        curves = {}

        for lbl in config['labels']:
            curves[lbl] = random.choice([
                np.cumsum(np.random.normal(0, 2, 50)) + random.randint(20, 80),
                np.sin(x_data / 10) * 20 + random.randint(40, 60),
                (x_data / 10) ** 2 + random.randint(10, 30)
            ])

        q, opts, ans, t_info = generate_qa(x_data, curves, config)
        img_path, line_px = create_precise_chart(run_id, x_data, curves, config, t_info)
        final_imgs = draw_infinite_line(img_path, line_px)

        ltrs = "ABCDEF"
        opt_str = "\n".join([f"{ltrs[j]}. {val}" for j, val in enumerate(opts)])
        correct_ltr = ltrs[opts.index(ans)]

        results.append({
            "id": run_id,
            "image": final_imgs,
            "line_pixel_coords": line_px,
            "conversations": [
                {"from": "human", "value": f"<image>\n{q}\n{opt_str}"},
                {"from": "gpt", "value": f"The correct answer is {correct_ltr}."}
            ],
            "answer": ans
        })

        if (i + 1) % 5 == 0:
            print(f"Progress: {i + 1}/{num_samples}")

    with open(os.path.join(OUTPUT_DIR, "dataset.json"), "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    run_pipeline(20)
