# 🧠 Neural Network GUI — Interactive Trainer

A fully-featured neural network trainer with a beautiful dark-themed GUI.
Built with Python, PyQt5, and Matplotlib — **no TensorFlow or PyTorch required!**

---

## 🚀 Quick Setup (VS Code)

### Step 1: Install Python
Make sure Python 3.8+ is installed: https://www.python.org/downloads/

### Step 2: Install Dependencies
Open a terminal in this folder and run:
```bash
pip install PyQt5 matplotlib numpy scikit-learn
```

### Step 3: Run the App
```bash
python neural_network_gui.py
```

Or in VS Code: Open `neural_network_gui.py` → Press **F5** or click ▶ Run

---

## 🎮 Features

| Feature | Description |
|---|---|
| **5 Datasets** | XOR, Circles, Moons, Blobs, Spiral |
| **Custom Architecture** | 1–5 hidden layers, 2–64 neurons each |
| **3 Activations** | ReLU, Sigmoid, Tanh |
| **Live Decision Boundary** | Updates every 5 epochs during training |
| **Loss & Accuracy Charts** | Real-time dual-axis graph |
| **Weight Heatmaps** | Visual inspection of all weight matrices |
| **Network Visualizer** | Animated architecture diagram with weights |
| **Training Log** | Live console output |
| **Stop/Reset** | Full control over training |

---

## 🎛️ How to Use

1. **Choose a Dataset** (e.g. Spiral) → Click "Generate Dataset"
2. **Set Architecture**: 2 hidden layers, 16 neurons, ReLU
3. **Set Training**: LR=0.01, Epochs=300, Batch=32
4. **Click ▶ Train** and watch it learn in real time!
5. Switch between the tabs to see Decision Boundary, Loss curves, and Weight heatmaps

---

## 💡 Experiment Ideas

- Try **XOR** with 1 hidden layer (8 neurons) — can it solve it?
- Try **Spiral** with 3 hidden layers (32 neurons) — needs more power!
- Compare **ReLU vs Sigmoid** activation on the same dataset
- Set learning rate to **0.5** — watch it diverge (loss explodes!)
- Set it to **0.001** — watch it converge slowly but steadily

---

## 📁 File Structure
```
neural_network_gui/
├── neural_network_gui.py   ← Main application (run this!)
├── requirements.txt        ← Python dependencies
└── README.md               ← This file
```

---

## 🐛 Troubleshooting

**PyQt5 not found?**
```bash
pip install --upgrade pip
pip install PyQt5
```

**matplotlib backend error on Linux?**
```bash
sudo apt-get install python3-pyqt5
```

**scikit-learn not found?**
```bash
pip install scikit-learn
```
