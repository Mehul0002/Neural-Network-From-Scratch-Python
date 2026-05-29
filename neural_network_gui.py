"""
Neural Network GUI - Full Featured Desktop Application
Requirements: pip install PyQt5 matplotlib numpy scikit-learn
Run: python neural_network_gui.py
"""

import sys
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QSlider, QComboBox, QGroupBox, QGridLayout,
    QTextEdit, QSplitter, QFrame, QSpinBox, QDoubleSpinBox, QProgressBar,
    QTabWidget, QCheckBox, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QPainter, QPen, QBrush, QColor, QFont, QLinearGradient
import matplotlib
matplotlib.use('Qt5Agg')
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import threading
import time


# ─────────────────────────────────────────────
#  Neural Network Implementation (from scratch)
# ─────────────────────────────────────────────

def sigmoid(x):
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

def sigmoid_deriv(x):
    s = sigmoid(x)
    return s * (1 - s)

def relu(x):
    return np.maximum(0, x)

def relu_deriv(x):
    return (x > 0).astype(float)

def tanh_act(x):
    return np.tanh(x)

def tanh_deriv(x):
    return 1 - np.tanh(x)**2

def softmax(x):
    e = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)

ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_deriv),
    'relu': (relu, relu_deriv),
    'tanh': (tanh_act, tanh_deriv),
}


class NeuralNetwork:
    def __init__(self, layer_sizes, activation='relu', learning_rate=0.01):
        self.layer_sizes = layer_sizes
        self.activation_name = activation
        self.lr = learning_rate
        self.act, self.act_d = ACTIVATIONS[activation]
        self.weights = []
        self.biases = []
        self.loss_history = []
        self.acc_history = []
        self._init_weights()

    def _init_weights(self):
        self.weights = []
        self.biases = []
        for i in range(len(self.layer_sizes) - 1):
            fan_in = self.layer_sizes[i]
            fan_out = self.layer_sizes[i+1]
            w = np.random.randn(fan_in, fan_out) * np.sqrt(2.0 / fan_in)
            b = np.zeros((1, fan_out))
            self.weights.append(w)
            self.biases.append(b)

    def forward(self, X):
        self.zs = []
        self.as_ = [X]
        a = X
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = a @ w + b
            self.zs.append(z)
            if i == len(self.weights) - 1:
                # output layer: softmax for multiclass, sigmoid for binary
                if self.layer_sizes[-1] > 1:
                    a = softmax(z)
                else:
                    a = sigmoid(z)
            else:
                a = self.act(z)
            self.as_.append(a)
        return self.as_[-1]

    def backward(self, X, y):
        m = X.shape[0]
        y_enc = self._encode_labels(y)
        out = self.as_[-1]
        delta = out - y_enc

        grads_w = []
        grads_b = []

        for i in reversed(range(len(self.weights))):
            dw = self.as_[i].T @ delta / m
            db = delta.mean(axis=0, keepdims=True)
            grads_w.insert(0, dw)
            grads_b.insert(0, db)
            if i > 0:
                delta = (delta @ self.weights[i].T) * self.act_d(self.zs[i-1])

        for i in range(len(self.weights)):
            self.weights[i] -= self.lr * grads_w[i]
            self.biases[i] -= self.lr * grads_b[i]

    def _encode_labels(self, y):
        n_classes = self.layer_sizes[-1]
        if n_classes == 1:
            return y.reshape(-1, 1).astype(float)
        m = y.shape[0]
        enc = np.zeros((m, n_classes))
        enc[np.arange(m), y.astype(int)] = 1
        return enc

    def compute_loss(self, X, y):
        out = self.forward(X)
        y_enc = self._encode_labels(y)
        eps = 1e-12
        return -np.mean(y_enc * np.log(out + eps))

    def predict(self, X):
        out = self.forward(X)
        if self.layer_sizes[-1] == 1:
            return (out >= 0.5).astype(int).flatten()
        return np.argmax(out, axis=1)

    def accuracy(self, X, y):
        return np.mean(self.predict(X) == y.astype(int))


# ─────────────────────────────────────────────
#  Dataset Generator
# ─────────────────────────────────────────────

def make_dataset(name, n=500, noise=0.1):
    np.random.seed(42)
    if name == 'XOR':
        X = np.random.randn(n, 2)
        y = (((X[:,0] > 0) & (X[:,1] > 0)) | ((X[:,0] < 0) & (X[:,1] < 0))).astype(int)
    elif name == 'Circles':
        from sklearn.datasets import make_circles
        X, y = make_circles(n_samples=n, noise=noise, factor=0.4)
    elif name == 'Moons':
        from sklearn.datasets import make_moons
        X, y = make_moons(n_samples=n, noise=noise)
    elif name == 'Blobs':
        from sklearn.datasets import make_blobs
        X, y = make_blobs(n_samples=n, centers=3, cluster_std=1.0)
    elif name == 'Spiral':
        def spiral(n, label, noise):
            ix = np.arange(n)
            r = ix / n * 1 + noise * np.random.randn(n)
            t = 1.25 * ix / n * 2 * np.pi + label * 4 + noise * np.random.randn(n)
            return np.column_stack([r*np.sin(t), r*np.cos(t)]), np.full(n, label)
        half = n // 2
        X1, y1 = spiral(half, 0, noise)
        X2, y2 = spiral(half, 1, noise)
        X, y = np.vstack([X1, X2]), np.hstack([y1, y2])
    else:
        X = np.random.randn(n, 2)
        y = (X[:,0] + X[:,1] > 0).astype(int)

    # Normalize
    X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return X, y.astype(int)


# ─────────────────────────────────────────────
#  Training Thread
# ─────────────────────────────────────────────

class TrainingThread(QThread):
    update_signal = pyqtSignal(int, float, float)
    finished_signal = pyqtSignal()
    log_signal = pyqtSignal(str)

    def __init__(self, nn, X, y, epochs, batch_size):
        super().__init__()
        self.nn = nn
        self.X = X
        self.y = y
        self.epochs = epochs
        self.batch_size = batch_size
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        m = self.X.shape[0]
        for epoch in range(1, self.epochs + 1):
            if not self._running:
                break
            # Mini-batch SGD
            idx = np.random.permutation(m)
            for start in range(0, m, self.batch_size):
                batch_idx = idx[start:start+self.batch_size]
                Xb, yb = self.X[batch_idx], self.y[batch_idx]
                self.nn.forward(Xb)
                self.nn.backward(Xb, yb)

            loss = self.nn.compute_loss(self.X, self.y)
            acc = self.nn.accuracy(self.X, self.y)
            self.nn.loss_history.append(loss)
            self.nn.acc_history.append(acc)
            self.update_signal.emit(epoch, loss, acc)

            if epoch % 10 == 0:
                self.log_signal.emit(f"Epoch {epoch:4d} | Loss: {loss:.4f} | Acc: {acc*100:.1f}%")

            time.sleep(0.01)  # Small delay so GUI updates smoothly

        self.finished_signal.emit()


# ─────────────────────────────────────────────
#  Network Visualizer Widget
# ─────────────────────────────────────────────

class NetworkVisualizerWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.layer_sizes = [2, 4, 4, 1]
        self.weights = None
        self.setMinimumHeight(280)

    def set_network(self, layer_sizes, weights=None):
        self.layer_sizes = layer_sizes
        self.weights = weights
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()

        # Background gradient
        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0, QColor(15, 15, 35))
        grad.setColorAt(1, QColor(25, 25, 50))
        painter.fillRect(0, 0, w, h, grad)

        layers = self.layer_sizes
        max_neurons = max(min(n, 8) for n in layers)
        n_layers = len(layers)
        margin_x = 60
        margin_y = 40
        layer_spacing = (w - 2*margin_x) / (n_layers - 1) if n_layers > 1 else w/2
        neuron_r = 14

        # Compute positions
        positions = []
        for li, n in enumerate(layers):
            display_n = min(n, 8)
            xs = margin_x + li * layer_spacing
            if display_n == 1:
                ys_list = [h / 2]
            else:
                total_h = (display_n - 1) * (h - 2*margin_y) / (max_neurons - 1) if max_neurons > 1 else 0
                start_y = h/2 - total_h/2
                step = total_h / (display_n - 1) if display_n > 1 else 0
                ys_list = [start_y + i*step for i in range(display_n)]
            positions.append((xs, ys_list))

        # Draw connections
        for li in range(len(layers) - 1):
            xs1, ys1 = positions[li]
            xs2, ys2 = positions[li+1]
            for j, y1 in enumerate(ys1):
                for k, y2 in enumerate(ys2):
                    if self.weights and li < len(self.weights):
                        w_val = float(np.clip(self.weights[li][j % self.weights[li].shape[0],
                                                               k % self.weights[li].shape[1]], -2, 2))
                        intensity = int(abs(w_val) / 2 * 180)
                        if w_val > 0:
                            color = QColor(80, intensity+40, 255, intensity+40)
                        else:
                            color = QColor(255, intensity+40, 80, intensity+40)
                    else:
                        color = QColor(80, 100, 180, 60)
                    pen = QPen(color, 1)
                    painter.setPen(pen)
                    painter.drawLine(int(xs1), int(y1), int(xs2), int(y2))

        # Draw neurons
        for li, (xs, ys_list) in enumerate(positions):
            n = layers[li]
            display_n = min(n, 8)
            for ni, y in enumerate(ys_list):
                # Glow effect
                for r_extra in [6, 4, 2]:
                    glow = QColor(100, 160, 255, 20)
                    painter.setPen(Qt.NoPen)
                    painter.setBrush(QBrush(glow))
                    painter.drawEllipse(int(xs - neuron_r - r_extra), int(y - neuron_r - r_extra),
                                        (neuron_r + r_extra)*2, (neuron_r + r_extra)*2)

                # Neuron fill
                grad2 = QLinearGradient(xs - neuron_r, y - neuron_r, xs + neuron_r, y + neuron_r)
                grad2.setColorAt(0, QColor(80, 160, 255))
                grad2.setColorAt(1, QColor(40, 80, 200))
                painter.setBrush(QBrush(grad2))
                painter.setPen(QPen(QColor(150, 200, 255), 1.5))
                painter.drawEllipse(int(xs - neuron_r), int(y - neuron_r), neuron_r*2, neuron_r*2)

            # Show "..." if neurons truncated
            if n > 8:
                painter.setPen(QPen(QColor(180, 200, 255)))
                painter.setFont(QFont('Arial', 9))
                painter.drawText(int(xs) - 8, int(ys_list[-1]) + 30, f"+{n-8}")

        # Layer labels
        label_names = ['Input'] + [f'Hidden {i}' for i in range(1, len(layers)-1)] + ['Output']
        for li, (xs, _) in enumerate(positions):
            painter.setPen(QPen(QColor(150, 200, 255)))
            painter.setFont(QFont('Consolas', 8))
            label = f"{label_names[li]}\n({layers[li]})"
            painter.drawText(int(xs) - 25, h - 22, 50, 20, Qt.AlignCenter, f"{layers[li]}")

        painter.end()


# ─────────────────────────────────────────────
#  Main Window
# ─────────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("🧠 Neural Network GUI — Interactive Trainer")
        self.setMinimumSize(1200, 800)
        self.nn = None
        self.X = None
        self.y = None
        self.train_thread = None
        self._init_ui()
        self._apply_dark_theme()
        self.generate_dataset()

    def _apply_dark_theme(self):
        self.setStyleSheet("""
            QMainWindow, QWidget { background-color: #0e0e1a; color: #c8d8ff; }
            QGroupBox {
                border: 1px solid #2a3060;
                border-radius: 8px;
                margin-top: 10px;
                font-weight: bold;
                color: #7aadff;
                padding: 8px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton {
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #2a4080, stop:1 #1a2860);
                color: #aaccff;
                border: 1px solid #3a50a0;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
            }
            QPushButton:hover { background: #3a5090; border-color: #6080d0; }
            QPushButton:pressed { background: #1a3060; }
            QPushButton:disabled { background: #1a1a2a; color: #4a4a6a; }
            QComboBox, QSpinBox, QDoubleSpinBox {
                background: #1a1a2e;
                border: 1px solid #2a3060;
                border-radius: 4px;
                padding: 4px 8px;
                color: #aaccff;
            }
            QSlider::groove:horizontal {
                height: 4px; background: #2a3060; border-radius: 2px;
            }
            QSlider::handle:horizontal {
                background: #5080e0; width: 16px; height: 16px;
                margin: -6px 0; border-radius: 8px;
            }
            QSlider::sub-page:horizontal { background: #4070d0; border-radius: 2px; }
            QTextEdit {
                background: #080812;
                border: 1px solid #1a2050;
                border-radius: 6px;
                color: #60ff90;
                font-family: Consolas, monospace;
                font-size: 11px;
            }
            QTabWidget::pane { border: 1px solid #2a3060; }
            QTabBar::tab {
                background: #0e0e1a;
                color: #6080b0;
                padding: 8px 16px;
                border: 1px solid #2a3060;
            }
            QTabBar::tab:selected { background: #1a2050; color: #aaccff; }
            QProgressBar {
                border: 1px solid #2a3060;
                border-radius: 4px;
                background: #080812;
                text-align: center;
                color: #aaccff;
            }
            QProgressBar::chunk { background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                stop:0 #2040a0, stop:1 #4080e0); border-radius: 3px; }
            QLabel { color: #aaccff; }
        """)

    def _init_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(12)
        main_layout.setContentsMargins(12, 12, 12, 12)

        # ── Left Panel: Controls ──────────────────────────
        left_panel = QWidget()
        left_panel.setFixedWidth(300)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setSpacing(8)

        # Title
        title = QLabel("🧠 Neural Network\nInteractive Trainer")
        title.setAlignment(Qt.AlignCenter)
        title.setFont(QFont('Consolas', 14, QFont.Bold))
        title.setStyleSheet("color: #7aadff; padding: 10px; border-bottom: 1px solid #2a3060;")
        left_layout.addWidget(title)

        # Dataset Group
        dataset_group = QGroupBox("📊 Dataset")
        dg_layout = QGridLayout(dataset_group)
        dg_layout.addWidget(QLabel("Type:"), 0, 0)
        self.dataset_combo = QComboBox()
        self.dataset_combo.addItems(['XOR', 'Circles', 'Moons', 'Blobs', 'Spiral'])
        dg_layout.addWidget(self.dataset_combo, 0, 1)
        dg_layout.addWidget(QLabel("Samples:"), 1, 0)
        self.samples_spin = QSpinBox()
        self.samples_spin.setRange(100, 2000)
        self.samples_spin.setValue(500)
        self.samples_spin.setSingleStep(100)
        dg_layout.addWidget(self.samples_spin, 1, 1)
        dg_layout.addWidget(QLabel("Noise:"), 2, 0)
        self.noise_spin = QDoubleSpinBox()
        self.noise_spin.setRange(0.0, 0.5)
        self.noise_spin.setValue(0.1)
        self.noise_spin.setSingleStep(0.05)
        dg_layout.addWidget(self.noise_spin, 2, 1)
        self.gen_btn = QPushButton("🔄 Generate Dataset")
        self.gen_btn.clicked.connect(self.generate_dataset)
        dg_layout.addWidget(self.gen_btn, 3, 0, 1, 2)
        left_layout.addWidget(dataset_group)

        # Architecture Group
        arch_group = QGroupBox("🏗️ Network Architecture")
        ag_layout = QGridLayout(arch_group)
        ag_layout.addWidget(QLabel("Hidden Layers:"), 0, 0)
        self.hidden_spin = QSpinBox()
        self.hidden_spin.setRange(1, 5)
        self.hidden_spin.setValue(2)
        ag_layout.addWidget(self.hidden_spin, 0, 1)
        ag_layout.addWidget(QLabel("Neurons/Layer:"), 1, 0)
        self.neurons_spin = QSpinBox()
        self.neurons_spin.setRange(2, 64)
        self.neurons_spin.setValue(8)
        ag_layout.addWidget(self.neurons_spin, 1, 1)
        ag_layout.addWidget(QLabel("Activation:"), 2, 0)
        self.act_combo = QComboBox()
        self.act_combo.addItems(['relu', 'sigmoid', 'tanh'])
        ag_layout.addWidget(self.act_combo, 2, 1)
        left_layout.addWidget(arch_group)

        # Training Group
        train_group = QGroupBox("⚙️ Training Parameters")
        tg_layout = QGridLayout(train_group)
        tg_layout.addWidget(QLabel("Learning Rate:"), 0, 0)
        self.lr_combo = QComboBox()
        self.lr_combo.addItems(['0.001', '0.005', '0.01', '0.05', '0.1', '0.5'])
        self.lr_combo.setCurrentIndex(2)
        tg_layout.addWidget(self.lr_combo, 0, 1)
        tg_layout.addWidget(QLabel("Epochs:"), 1, 0)
        self.epochs_spin = QSpinBox()
        self.epochs_spin.setRange(10, 2000)
        self.epochs_spin.setValue(200)
        self.epochs_spin.setSingleStep(50)
        tg_layout.addWidget(self.epochs_spin, 1, 1)
        tg_layout.addWidget(QLabel("Batch Size:"), 2, 0)
        self.batch_combo = QComboBox()
        self.batch_combo.addItems(['16', '32', '64', '128', 'Full'])
        self.batch_combo.setCurrentIndex(1)
        tg_layout.addWidget(self.batch_combo, 2, 1)
        left_layout.addWidget(train_group)

        # Train/Stop buttons
        btn_layout = QHBoxLayout()
        self.train_btn = QPushButton("▶ Train")
        self.train_btn.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #1a6030,stop:1 #0a4020); border-color: #2a8040; color: #80ffaa;")
        self.train_btn.clicked.connect(self.start_training)
        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.setStyleSheet("background: qlineargradient(x1:0,y1:0,x2:0,y2:1,stop:0 #601010,stop:1 #400808); border-color: #902020; color: #ffaaaa;")
        self.stop_btn.setEnabled(False)
        self.stop_btn.clicked.connect(self.stop_training)
        btn_layout.addWidget(self.train_btn)
        btn_layout.addWidget(self.stop_btn)
        left_layout.addLayout(btn_layout)

        self.reset_btn = QPushButton("🔁 Reset Network")
        self.reset_btn.clicked.connect(self.reset_network)
        left_layout.addWidget(self.reset_btn)

        # Progress & Stats
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        left_layout.addWidget(self.progress_bar)

        stats_group = QGroupBox("📈 Live Stats")
        sg_layout = QGridLayout(stats_group)
        self.epoch_label = QLabel("Epoch: —")
        self.loss_label = QLabel("Loss: —")
        self.acc_label = QLabel("Accuracy: —")
        for lbl in [self.epoch_label, self.loss_label, self.acc_label]:
            lbl.setFont(QFont('Consolas', 11, QFont.Bold))
            lbl.setStyleSheet("color: #60ffaa;")
        sg_layout.addWidget(self.epoch_label, 0, 0)
        sg_layout.addWidget(self.loss_label, 1, 0)
        sg_layout.addWidget(self.acc_label, 2, 0)
        left_layout.addWidget(stats_group)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # ── Right Panel: Visualizations ───────────────────
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setSpacing(8)

        # Network visualizer
        net_group = QGroupBox("🔮 Network Architecture Visualizer")
        net_g_layout = QVBoxLayout(net_group)
        self.net_viz = NetworkVisualizerWidget()
        self.net_viz.setFixedHeight(180)
        net_g_layout.addWidget(self.net_viz)
        right_layout.addWidget(net_group)

        # Tabs for charts
        self.tabs = QTabWidget()

        # Tab 1: Data + Decision Boundary
        boundary_tab = QWidget()
        bl = QVBoxLayout(boundary_tab)
        self.fig_boundary = Figure(figsize=(8, 4), facecolor='#0a0a1a')
        self.canvas_boundary = FigureCanvas(self.fig_boundary)
        bl.addWidget(self.canvas_boundary)
        self.tabs.addTab(boundary_tab, "🗺️ Decision Boundary")

        # Tab 2: Loss & Accuracy
        loss_tab = QWidget()
        ll = QVBoxLayout(loss_tab)
        self.fig_loss = Figure(figsize=(8, 4), facecolor='#0a0a1a')
        self.canvas_loss = FigureCanvas(self.fig_loss)
        ll.addWidget(self.canvas_loss)
        self.tabs.addTab(loss_tab, "📉 Loss & Accuracy")

        # Tab 3: Weight Heatmap
        weights_tab = QWidget()
        wl = QVBoxLayout(weights_tab)
        self.fig_weights = Figure(figsize=(8, 4), facecolor='#0a0a1a')
        self.canvas_weights = FigureCanvas(self.fig_weights)
        wl.addWidget(self.canvas_weights)
        self.tabs.addTab(weights_tab, "🌡️ Weight Heatmap")

        right_layout.addWidget(self.tabs)

        # Log console
        log_group = QGroupBox("💻 Training Log")
        log_layout = QVBoxLayout(log_group)
        self.log_text = QTextEdit()
        self.log_text.setMaximumHeight(120)
        self.log_text.setReadOnly(True)
        log_layout.addWidget(self.log_text)
        right_layout.addWidget(log_group)

        main_layout.addWidget(right_panel)

    # ─────────────────────────────────────────
    #  Actions
    # ─────────────────────────────────────────

    def generate_dataset(self):
        name = self.dataset_combo.currentText()
        n = self.samples_spin.value()
        noise = self.noise_spin.value()
        self.X, self.y = make_dataset(name, n, noise)
        self.log(f"✅ Generated '{name}' dataset — {n} samples, noise={noise}")
        self.plot_boundary()

    def build_network(self):
        hidden = self.hidden_spin.value()
        neurons = self.neurons_spin.value()
        n_classes = len(np.unique(self.y))
        out_size = n_classes if n_classes > 2 else 1
        layer_sizes = [self.X.shape[1]] + [neurons]*hidden + [out_size]
        lr = float(self.lr_combo.currentText())
        activation = self.act_combo.currentText()
        self.nn = NeuralNetwork(layer_sizes, activation, lr)
        self.net_viz.set_network(layer_sizes)
        return layer_sizes

    def start_training(self):
        if self.X is None:
            self.generate_dataset()
        self.build_network()
        self.nn.loss_history.clear()
        self.nn.acc_history.clear()

        epochs = self.epochs_spin.value()
        batch_str = self.batch_combo.currentText()
        batch = len(self.X) if batch_str == 'Full' else int(batch_str)

        self.train_thread = TrainingThread(self.nn, self.X, self.y, epochs, batch)
        self.train_thread.update_signal.connect(self.on_epoch)
        self.train_thread.finished_signal.connect(self.on_training_done)
        self.train_thread.log_signal.connect(self.log)
        self.train_thread.start()

        self.train_btn.setEnabled(False)
        self.stop_btn.setEnabled(True)
        self.progress_bar.setRange(0, epochs)
        self.log(f"🚀 Training started | Epochs={epochs} | Batch={batch_str} | LR={self.nn.lr}")

    def stop_training(self):
        if self.train_thread:
            self.train_thread.stop()
        self.log("⏹ Training stopped by user.")

    def reset_network(self):
        if self.train_thread and self.train_thread.isRunning():
            self.train_thread.stop()
        self.nn = None
        self.epoch_label.setText("Epoch: —")
        self.loss_label.setText("Loss: —")
        self.acc_label.setText("Accuracy: —")
        self.progress_bar.setValue(0)
        self.log("🔁 Network reset.")
        self.plot_boundary()

    def on_epoch(self, epoch, loss, acc):
        self.epoch_label.setText(f"Epoch: {epoch}")
        self.loss_label.setText(f"Loss: {loss:.4f}")
        self.acc_label.setText(f"Accuracy: {acc*100:.1f}%")
        self.progress_bar.setValue(epoch)
        # Update plots every 5 epochs
        if epoch % 5 == 0 or epoch == 1:
            self.plot_boundary()
            self.plot_loss()
            self.plot_weights()
            self.net_viz.set_network(self.nn.layer_sizes, self.nn.weights)

    def on_training_done(self):
        self.train_btn.setEnabled(True)
        self.stop_btn.setEnabled(False)
        self.plot_boundary()
        self.plot_loss()
        self.plot_weights()
        self.log("✅ Training complete!")

    def log(self, msg):
        self.log_text.append(msg)
        self.log_text.verticalScrollBar().setValue(
            self.log_text.verticalScrollBar().maximum())

    # ─────────────────────────────────────────
    #  Plots
    # ─────────────────────────────────────────

    def plot_boundary(self):
        if self.X is None:
            return
        fig = self.fig_boundary
        fig.clear()
        ax = fig.add_subplot(111)
        ax.set_facecolor('#0a0a1a')
        fig.patch.set_facecolor('#0a0a1a')

        X, y = self.X, self.y

        if self.nn is not None:
            h = 0.04
            x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
            y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
            xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                                  np.arange(y_min, y_max, h))
            grid = np.c_[xx.ravel(), yy.ravel()]
            Z = self.nn.predict(grid).reshape(xx.shape)
            n_classes = len(np.unique(y))
            cmap_bg = plt.cm.RdYlBu if n_classes <= 2 else plt.cm.tab10
            ax.contourf(xx, yy, Z, alpha=0.35, cmap=cmap_bg)

        colors = plt.cm.tab10(np.linspace(0, 0.5, len(np.unique(y))))
        for ci, cls in enumerate(np.unique(y)):
            mask = y == cls
            ax.scatter(X[mask, 0], X[mask, 1], s=20, alpha=0.8,
                       c=[matplotlib.colors.to_hex(colors[ci])],
                       edgecolors='none', label=f'Class {cls}')

        ax.set_title("Decision Boundary" if self.nn else "Dataset",
                     color='#7aadff', fontsize=12)
        ax.tick_params(colors='#5060a0')
        for spine in ax.spines.values():
            spine.set_edgecolor('#1a2050')
        ax.legend(facecolor='#0a0a1a', labelcolor='#aaccff', framealpha=0.7, fontsize=8)
        self.canvas_boundary.draw()

    def plot_loss(self):
        if self.nn is None or not self.nn.loss_history:
            return
        fig = self.fig_loss
        fig.clear()
        ax1 = fig.add_subplot(111)
        ax2 = ax1.twinx()
        ax1.set_facecolor('#0a0a1a')
        fig.patch.set_facecolor('#0a0a1a')

        epochs = range(1, len(self.nn.loss_history)+1)
        ax1.plot(epochs, self.nn.loss_history, color='#ff6060', linewidth=1.5, label='Loss')
        ax2.plot(epochs, [a*100 for a in self.nn.acc_history],
                 color='#60ffaa', linewidth=1.5, label='Accuracy %')

        ax1.set_xlabel('Epoch', color='#5060a0')
        ax1.set_ylabel('Loss', color='#ff6060')
        ax2.set_ylabel('Accuracy (%)', color='#60ffaa')
        ax1.set_title("Training Loss & Accuracy", color='#7aadff', fontsize=12)
        ax1.tick_params(colors='#5060a0')
        ax2.tick_params(colors='#5060a0')
        for spine in ax1.spines.values():
            spine.set_edgecolor('#1a2050')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1+lines2, labels1+labels2, facecolor='#0a0a1a',
                   labelcolor='#aaccff', framealpha=0.7, fontsize=9)
        self.canvas_loss.draw()

    def plot_weights(self):
        if self.nn is None:
            return
        fig = self.fig_weights
        fig.clear()
        n = len(self.nn.weights)
        cols = min(n, 3)
        rows = (n + cols - 1) // cols
        fig.patch.set_facecolor('#0a0a1a')

        for i, w in enumerate(self.nn.weights):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.set_facecolor('#0a0a1a')
            disp_w = w[:min(16, w.shape[0]), :min(16, w.shape[1])]
            im = ax.imshow(disp_w, cmap='RdBu_r', aspect='auto',
                           vmin=-2, vmax=2)
            ax.set_title(f"W{i+1}: {w.shape[0]}→{w.shape[1]}",
                         color='#7aadff', fontsize=9)
            ax.tick_params(colors='#5060a0', labelsize=7)
            fig.colorbar(im, ax=ax, fraction=0.046).ax.tick_params(
                colors='#5060a0', labelsize=7)

        fig.suptitle("Weight Heatmaps (first 16×16)", color='#7aadff', fontsize=11)
        fig.tight_layout()
        self.canvas_weights.draw()


# ─────────────────────────────────────────────
#  Entry Point
# ─────────────────────────────────────────────

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
