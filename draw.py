import tkinter as tk
import struct
import math
import numpy as np

# ── Load weights from our C binary format ──────────────
def load_weights(path):
    with open(path, 'rb') as f:
        def read_matrix(rows, cols):
            data = struct.unpack('d' * rows * cols, f.read(8 * rows * cols))
            return [list(data[i*cols:(i+1)*cols]) for i in range(rows)]
        def read_vector(n):
            return list(struct.unpack('d' * n, f.read(8 * n)))

        W1 = read_matrix(784, 128)
        W2 = read_matrix(128, 64)
        W3 = read_matrix(64,  10)
        b1 = read_vector(128)
        b2 = read_vector(64)
        b3 = read_vector(10)
    return W1, W2, W3, b1, b2, b3

# ── Activations ────────────────────────────────────────
def relu(x):
    return max(0.0, x)

def softmax(x):
    m = max(x)
    e = [math.exp(v - m) for v in x]
    s = sum(e)
    return [v / s for v in e]

# ── Forward pass ───────────────────────────────────────
def forward(pixels, W1, W2, W3, b1, b2, b3):
    # Hidden layer 1
    a1 = [relu(sum(pixels[i] * W1[i][j] for i in range(784)) + b1[j])
          for j in range(128)]
    # Hidden layer 2
    a2 = [relu(sum(a1[i] * W2[i][j] for i in range(128)) + b2[j])
          for j in range(64)]
    # Output layer
    z3 = [sum(a2[i] * W3[i][j] for i in range(64)) + b3[j]
          for j in range(10)]
    return softmax(z3)

# ── Load weights once ──────────────────────────────────
print("Loading weights...")
W1, W2, W3, b1, b2, b3 = load_weights("weights/mnist.dat")
print("Ready.")

# ── GUI ────────────────────────────────────────────────
CANVAS_SIZE = 280   # 280x280 display canvas
GRID_SIZE   = 28    # 28x28 network input

grid = [[0.0] * GRID_SIZE for _ in range(GRID_SIZE)]

root = tk.Tk()
root.title("Draw a digit")
root.resizable(False, False)

canvas = tk.Canvas(root, width=CANVAS_SIZE, height=CANVAS_SIZE, bg='black', cursor='crosshair')
canvas.pack()

# Digit label
result_var = tk.StringVar(value="Draw a digit")
result_lbl = tk.Label(root, textvariable=result_var, font=("Helvetica", 28, "bold"), fg="white", bg="#1a1a1a")
result_lbl.pack(fill=tk.X)

# Confidence bars
bar_frame = tk.Frame(root, bg="#1a1a1a")
bar_frame.pack(fill=tk.X, padx=10, pady=5)

bar_labels = []
bar_canvas = []
for d in range(10):
    row = tk.Frame(bar_frame, bg="#1a1a1a")
    row.pack(fill=tk.X, pady=1)
    tk.Label(row, text=str(d), width=2, bg="#1a1a1a", fg="white", font=("Helvetica", 11)).pack(side=tk.LEFT)
    bc = tk.Canvas(row, height=16, bg="#333333", highlightthickness=0)
    bc.pack(side=tk.LEFT, fill=tk.X, expand=True)
    bar_canvas.append(bc)

clear_btn = tk.Button(root, text="Clear", font=("Helvetica", 14),
                      bg="#444", fg="white", relief=tk.FLAT,
                      command=lambda: clear())
clear_btn.pack(fill=tk.X, padx=10, pady=8)

root.configure(bg="#1a1a1a")

# ── Drawing logic ──────────────────────────────────────
drawing = False
BRUSH   = 1.5   # brush radius in grid cells

def draw(event):
    gx = event.x / CANVAS_SIZE * GRID_SIZE
    gy = event.y / CANVAS_SIZE * GRID_SIZE
    r  = int(BRUSH) + 1
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            dist = math.sqrt((col + 0.5 - gx)**2 + (row + 0.5 - gy)**2)
            if dist < BRUSH:
                grid[row][col] = min(1.0, grid[row][col] + (1.0 - dist / BRUSH))
    redraw_canvas()
    predict()

def redraw_canvas():
    canvas.delete("all")
    cell = CANVAS_SIZE // GRID_SIZE
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            v = int(grid[row][col] * 255)
            color = f'#{v:02x}{v:02x}{v:02x}'
            x0, y0 = col * cell, row * cell
            canvas.create_rectangle(x0, y0, x0+cell, y0+cell, fill=color, outline='')

def clear():
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            grid[row][col] = 0.0
    redraw_canvas()
    result_var.set("Draw a digit")
    for bc in bar_canvas:
        bc.delete("all")

# ── Prediction ─────────────────────────────────────────
def predict():
    pixels = [grid[r][c] for r in range(GRID_SIZE) for c in range(GRID_SIZE)]
    if max(pixels) < 0.05:
        return
    probs = forward(pixels, W1, W2, W3, b1, b2, b3)
    pred  = probs.index(max(probs))
    result_var.set(f"Prediction: {pred}  ({probs[pred]*100:.1f}%)")

    # Update bars
    root.update_idletasks()
    for d in range(10):
        bc  = bar_canvas[d]
        w   = bc.winfo_width()
        bc.delete("all")
        fill_w = int(probs[d] * w)
        color  = "#1D9E75" if d == pred else "#555555"
        if fill_w > 0:
            bc.create_rectangle(0, 0, fill_w, 16, fill=color, outline='')
        bc.create_text(w - 4, 8, text=f"{probs[d]*100:.1f}%",
                       anchor='e', fill='white', font=("Helvetica", 10))

canvas.bind("<Button-1>",        lambda e: draw(e))
canvas.bind("<B1-Motion>",       lambda e: draw(e))

root.mainloop()
