"""Interactive Tkinter walkthrough of the trained SNN on a random MNIST test digit.

Steps: raw image -> DoG retina filter -> convolution of one filter (animated
sliding window) -> input current of all 16 feature maps -> spikes over time
(integrate-and-fire with WTA) -> output-layer voltage accumulation and prediction.

Requires the weights in ``weights/`` (``python scripts/train_deep_snn.py``) and a
display. Example:
    python scripts/snn_visualizer_gui.py
"""

import argparse
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle

from snn_detection import WEIGHTS_DIR
from snn_detection.data import load_mnist
from snn_detection.deep_snn import extract_patches
from snn_detection.encoding import dog_filter

BG_COLOR = "#0e0e0e"
PANEL_COLOR = "#222222"
ACCENT = "#00ffcc"
TEXT_FONT = ("Arial", 12)
N_FILTERS, MAP_SIDE, N_LOCATIONS = 16, 24, 576
STEPS = [
    "Step 1: raw MNIST input image",
    "Step 2: retina (DoG) filter -> 784 input neurons",
    "Step 3: convolution (sliding window) for one filter",
    "Step 4: total input current to the hidden layer (16 feature maps)",
    "Step 5: spikes over the time window (integrate & fire + WTA)",
    "Step 6: voltage accumulation in the output layer (prediction)",
]


class SNNModel:
    """Temporal inference with the trained weights.

    Each location integrates its input current for ``time_steps`` steps; the first
    filter to cross the threshold spikes (WTA) and the location is then inhibited
    for the rest of the window, so - as in training - at most one filter spikes per
    location. (Letting locations fire repeatedly drops accuracy to ~28%, because the
    classifier was trained on one-winner-per-location spike maps.)
    """

    def __init__(self, weights_dir=WEIGHTS_DIR, time_steps=20, threshold=2.0):
        self.time_steps, self.threshold = time_steps, threshold
        self.encoder_weights = np.load(Path(weights_dir) / "encoder_weights.npy")
        self.classifier_weights = np.load(Path(weights_dir) / "classifier_weights.npy")
        _, _, x_test, y_test = load_mnist()
        self.x_test, self.y_test = x_test, y_test

    def run_inference(self, idx=None):
        idx = np.random.randint(len(self.x_test)) if idx is None else idx
        self.image, self.label = self.x_test[idx], self.y_test[idx]
        self.dog_image = dog_filter(self.image)
        self.input_current = self.encoder_weights @ extract_patches(self.dog_image).T  # (16, 576)

        self.spikes_t = np.zeros((self.time_steps, N_FILTERS, N_LOCATIONS))
        v = np.zeros_like(self.input_current)
        fired = np.zeros(N_LOCATIONS, dtype=bool)
        for t in range(self.time_steps):
            v += self.input_current
            v[:, fired] = 0.0
            spiked = (v == v.max(axis=0)) & (v >= self.threshold)
            self.spikes_t[t] = spiked
            fired |= spiked.any(axis=0)

        self.output_v_t = np.cumsum([self.classifier_weights @ s.ravel() for s in self.spikes_t], axis=0)
        self.prediction = int(np.argmax(self.output_v_t[-1]))


class SNNVisualizerApp:
    def __init__(self, root, model):
        self.root, self.model = root, model
        self.root.title("Spiking neural network visualizer")
        self.root.geometry("1200x850")
        self.root.configure(bg=BG_COLOR)
        self.current_step = 0
        self.autoplay = False
        self.autoplay_ms = 2000
        self._configure_styles()
        self._build_ui()
        self.load_new_image()

    def _configure_styles(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TButton", font=TEXT_FONT, padding=6, background="#333333", foreground="white", borderwidth=0)
        style.map("TButton", background=[("active", "#555555"), ("disabled", "#222222")])
        style.configure("Control.TFrame", background=PANEL_COLOR)
        style.configure("TLabel", background=PANEL_COLOR, foreground="white", font=TEXT_FONT)
        style.configure("Info.TLabel", foreground=ACCENT, font=("Arial", 13, "bold"))

    def _build_ui(self):
        controls = ttk.Frame(self.root, padding="10", style="Control.TFrame")
        controls.pack(side=tk.TOP, fill=tk.X)
        top = ttk.Frame(controls, style="Control.TFrame")
        top.pack(side=tk.TOP, fill=tk.X, pady=(0, 5))

        self.btn_new = ttk.Button(top, text="New random image ↻", command=self.load_new_image)
        self.btn_new.pack(side=tk.LEFT, padx=10)
        self.btn_play = ttk.Button(top, text="Autoplay ▶", command=self.toggle_autoplay)
        self.btn_play.pack(side=tk.LEFT, padx=10)
        self.btn_next = ttk.Button(top, text="Next step ⏭", command=self.next_step)
        self.btn_next.pack(side=tk.LEFT, padx=10)
        ttk.Label(top, text="Seconds per step:").pack(side=tk.LEFT, padx=(20, 5))
        self.speed = ttk.Combobox(top, values=[0.5, 1, 2, 3], width=4, state="readonly")
        self.speed.set(2)
        self.speed.pack(side=tk.LEFT)
        self.speed.bind("<<ComboboxSelected>>", self.change_speed)
        self.lbl_info = ttk.Label(top, text="", style="Info.TLabel")
        self.lbl_info.pack(side=tk.RIGHT, padx=20)

        bottom = ttk.Frame(controls, style="Control.TFrame")
        bottom.pack(side=tk.TOP, fill=tk.X)
        self.lbl_slider = ttk.Label(bottom, text="Slider inactive")
        self.lbl_slider.pack(side=tk.LEFT, padx=(10, 5))
        self.t_var = tk.IntVar(value=0)
        self.slider = ttk.Scale(bottom, from_=0, to=1, variable=self.t_var, orient=tk.HORIZONTAL,
                                length=400, command=self.on_slider)
        self.slider.pack(side=tk.LEFT, padx=5)

        self.fig = plt.Figure(figsize=(12, 8), dpi=100)
        self.fig.patch.set_facecolor(BG_COLOR)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=1)

    # --- slider: window position in step 3, time in steps 5-6 -----------------
    def _slider_max(self):
        if self.current_step == 2:
            return N_LOCATIONS - 1
        if self.current_step in (4, 5):
            return self.model.time_steps - 1
        return None

    def _update_slider_label(self):
        t = self.t_var.get()
        if self.current_step == 2:
            self.lbl_slider.config(text=f"Convolution window position ({t}/{N_LOCATIONS - 1}):")
        elif self.current_step in (4, 5):
            self.lbl_slider.config(text=f"Time step (t={t}):")
        else:
            self.lbl_slider.config(text="Slider inactive")

    def setup_slider(self, value=0):
        maximum = self._slider_max()
        if maximum is None:
            self.slider.state(["disabled"])
            self.t_var.set(0)
        else:
            self.slider.configure(to=maximum)
            self.slider.state(["!disabled"])
            self.t_var.set(min(value, maximum))
        self._update_slider_label()

    def on_slider(self, _value=None):
        self._update_slider_label()
        if self._slider_max() is not None:
            self.update_plot()

    # --- navigation ------------------------------------------------------------
    def load_new_image(self):
        self.stop_autoplay()
        self.model.run_inference()
        self.current_step = 0
        self.setup_slider()
        self.update_plot()

    def next_step(self):
        if self.current_step < len(STEPS) - 1:
            self.current_step += 1
            self.setup_slider(value=10 ** 6)  # jump to the end of the animation
            self.update_plot()

    def change_speed(self, _event=None):
        self.autoplay_ms = int(float(self.speed.get()) * 1000)

    def toggle_autoplay(self):
        if self.autoplay:
            self.stop_autoplay()
        else:
            self.autoplay = True
            self.btn_play.config(text="Pause ⏸")
            self.btn_next.state(["disabled"])
            self.btn_new.state(["disabled"])
            self.root.after(self.autoplay_ms, self.auto_step)

    def stop_autoplay(self):
        self.autoplay = False
        self.btn_play.config(text="Autoplay ▶")
        self.btn_next.state(["!disabled"])
        self.btn_new.state(["!disabled"])

    def auto_step(self):
        if not self.autoplay:
            return
        maximum, t = self._slider_max(), self.t_var.get()
        if maximum is not None and t < maximum:
            # Animate within the step: 15 window positions or 1 time step per frame.
            self.t_var.set(min(t + (15 if self.current_step == 2 else 1), maximum))
            self.on_slider()
            self.root.after(40 if self.current_step == 2 else 150, self.auto_step)
        elif self.current_step < len(STEPS) - 1:
            self.current_step += 1
            self.setup_slider()
            self.update_plot()
            self.root.after(self.autoplay_ms, self.auto_step)
        else:
            self.stop_autoplay()

    # --- drawing ---------------------------------------------------------------
    def update_plot(self):
        m, step, t = self.model, self.current_step, self.t_var.get()
        self.fig.clf()
        self.lbl_info.config(text=f"[{step + 1}/{len(STEPS)}] {STEPS[step]}")
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1, 2.5], wspace=0.1)
        ax_img, ax_net = self.fig.add_subplot(gs[0]), self.fig.add_subplot(gs[1])
        for ax in (ax_img, ax_net):
            ax.set_facecolor(BG_COLOR)
            ax.axis("off")

        x_in, x_hid, x_out = 0.0, 1.5, 3.0
        y_in, y_hid, y_out = np.linspace(1, -1, 784), np.linspace(0.85, -0.85, N_FILTERS), np.linspace(0.6, -0.6, 10)
        ax_net.set_xlim(-0.2, 3.5)
        ax_net.set_ylim(-1.1, 1.1)
        c_in, c_hid, c_out = np.zeros(784), np.zeros(N_FILTERS), np.zeros(10)

        def square_image(image, box, anchor="C", **kwargs):
            """Draw ``image`` inside ``box = [x0, y0, width, height]`` (data units of ``ax_net``).

            The image gets its own inset axes with equal aspect, so MNIST-derived images
            stay square (letterboxed into the box) whatever the window size.
            """
            inset = ax_net.inset_axes(box, transform=ax_net.transData, zorder=5)
            inset.imshow(image, aspect="equal", **kwargs)
            inset.set_anchor(anchor)
            inset.axis("off")
            return inset

        def hidden_image(i, image, **kwargs):
            return square_image(image, [x_hid - 0.35, y_hid[i] - 0.045, 0.25, 0.09], anchor="E", **kwargs)

        if step == 0:
            ax_img.imshow(m.image, cmap="gray", aspect="equal")
            ax_img.set_title(f"Label: {m.label}", color="white", fontsize=18, pad=15)
        else:
            ax_img.imshow(m.dog_image, cmap="gray", aspect="equal")
            ax_img.set_title("DoG filtered", color="white", fontsize=18, pad=15)
            c_in = m.dog_image.flatten()

        if step == 2:
            row, col = divmod(t, MAP_SIDE)
            ax_img.add_patch(Rectangle((col - 0.5, row - 0.5), 5, 5, fill=False, edgecolor=ACCENT, linewidth=3))
            ax_net.text(1.65, 0.8, "Convolution (feature extraction)", color=ACCENT, ha="center",
                        fontsize=20, fontweight="bold")
            square_image(m.dog_image[row:row + 5, col:col + 5], [0.2, -0.3, 0.7, 0.7], cmap="gray")
            ax_net.text(0.55, 0.5, "Image patch (5x5)", color="white", ha="center", fontsize=16)
            ax_net.text(1.15, 0.05, "×", color="white", ha="center", fontsize=40, fontweight="bold")
            square_image(m.encoder_weights[0].reshape(5, 5), [1.4, -0.3, 0.7, 0.7], cmap="viridis")
            ax_net.text(1.75, 0.5, "Filter 0 weights", color="white", ha="center", fontsize=16)
            ax_net.text(2.35, 0.05, "=", color="white", ha="center", fontsize=40, fontweight="bold")
            partial = np.zeros(N_LOCATIONS)
            partial[:t + 1] = m.input_current[0, :t + 1]
            feature_map = square_image(partial.reshape(MAP_SIDE, MAP_SIDE), [2.6, -0.4, 0.8, 0.8], cmap="magma",
                                       vmin=0, vmax=m.input_current.max() + 1e-9)
            ax_net.text(3.0, 0.5, "Feature map (24x24)", color="white", ha="center", fontsize=16)
            # Highlight the output pixel computed for the current window position.
            feature_map.scatter([col], [row], color=ACCENT, s=80, zorder=6, edgecolors="white", linewidths=2)
            self.fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
            self.canvas.draw()
            return

        # Background connections of the network graph.
        rng = np.random.default_rng(42)
        for i, h in zip(rng.choice(784, 150, replace=False), rng.choice(N_FILTERS, 150)):
            ax_net.plot([x_in, x_hid], [y_in[i], y_hid[h]], color="white", alpha=0.04, linewidth=0.5, zorder=1)
        for h in range(N_FILTERS):
            for o in range(10):
                ax_net.plot([x_hid, x_out], [y_hid[h], y_out[o]], color="white", alpha=0.08, linewidth=1, zorder=1)

        if step == 1:
            ax_net.text(x_hid - 0.2, 1.05, "Feature weights", color=ACCENT, ha="center", fontsize=14,
                        fontweight="bold")
            for i in range(N_FILTERS):
                hidden_image(i, m.encoder_weights[i].reshape(5, 5), cmap="viridis")
        elif step == 3:
            v_max = m.input_current.max() + 1e-9
            ax_net.text(x_hid - 0.2, 1.05, "Total input current (24x24)", color="orange", ha="center",
                        fontsize=14, fontweight="bold")
            for i in range(N_FILTERS):
                current_map = m.input_current[i].reshape(MAP_SIDE, MAP_SIDE)
                hidden_image(i, current_map, cmap="magma", vmin=0, vmax=v_max)
                c_hid[i] = current_map.mean() / v_max
        elif step in (4, 5):
            header = f"Spikes at t={t}" if step == 4 else f"Integration up to t={t}"
            ax_net.text(x_hid - 0.2, 1.05, header, color="white" if step == 4 else ACCENT, ha="center",
                        fontsize=14, fontweight="bold")
            for i in range(N_FILTERS):
                spike_map = m.spikes_t[t, i].reshape(MAP_SIDE, MAP_SIDE)
                hidden_image(i, spike_map, cmap="gray", vmin=0, vmax=1)
                c_hid[i] = spike_map.max() * 2.0
            if step == 5:
                v_t = m.output_v_t[t]
                c_out = np.clip(v_t / (m.output_v_t.max() + 1e-9), 0, 1)
                pred_t = int(np.argmax(v_t))
                color = "#00ff00" if pred_t == m.label else "#ff0033"
                ax_net.text(x_hid, 1.15, f"True: {m.label}  |  Pred(t={t}): {pred_t}", color=color,
                            ha="center", fontsize=20, fontweight="bold")

        ax_net.scatter([x_in] * 784, y_in, c=c_in, cmap="gray", vmin=0, vmax=1, s=4, zorder=4)
        ax_net.scatter([x_hid] * N_FILTERS, y_hid, c=np.clip(c_hid, 0, 1), cmap="magma", vmin=0, vmax=1, s=150,
                       edgecolors="#444444", linewidth=1.5, zorder=4)
        if step == 5:
            winner = int(np.argmax(m.output_v_t[t]))
            colors = plt.cm.magma(c_out)
            for i in range(10):
                if i == winner:
                    border = "#00ff00" if i == m.label else "#ff0033"
                    ax_net.scatter(x_out, y_out[i], color=border, s=300, edgecolors="white", linewidth=2, zorder=6)
                else:
                    ax_net.scatter(x_out, y_out[i], color=colors[i], s=150, edgecolors="#444444", linewidth=1.5,
                                   zorder=4)
                ax_net.text(x_out + 0.15, y_out[i], str(i), color="white", fontsize=18, va="center",
                            fontweight="bold")
        else:
            ax_net.scatter([x_out] * 10, y_out, c=c_out, cmap="magma", vmin=0, vmax=1, s=150,
                           edgecolors="#444444", linewidth=1.5, zorder=4)
            for i in range(10):
                ax_net.text(x_out + 0.15, y_out[i], str(i), color="gray", fontsize=18, va="center",
                            fontweight="bold")
        self.fig.subplots_adjust(left=0.02, right=0.98, top=0.93, bottom=0.02)
        self.canvas.draw()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--weights-dir", type=Path, default=WEIGHTS_DIR)
    parser.add_argument("--time-steps", type=int, default=20, help="length of the simulated time window")
    args = parser.parse_args()

    root = tk.Tk()
    try:
        model = SNNModel(args.weights_dir, time_steps=args.time_steps)
    except FileNotFoundError:
        root.withdraw()
        messagebox.showerror("Missing weights", "No weights found in weights/. "
                             "Run: python scripts/train_deep_snn.py")
        root.destroy()
        return
    SNNVisualizerApp(root, model)
    root.mainloop()


if __name__ == "__main__":
    main()
