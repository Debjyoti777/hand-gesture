import tkinter as tk
from tkinter import font
import threading
import real_time_detection

# ------------------ MAIN WINDOW ------------------
root = tk.Tk()
root.title("Gesture Application")
root.geometry("700x500")
root.resizable(False, False)
root.configure(bg="#0f172a")  # dark blue background

# ------------------ FONTS ------------------
title_font = font.Font(family="Segoe UI", size=22, weight="bold")
subtitle_font = font.Font(family="Segoe UI", size=11)
btn_font = font.Font(family="Segoe UI", size=13, weight="bold")
status_font = font.Font(family="Segoe UI", size=10, slant="italic")

# ------------------ TOP FRAME ------------------
top_frame = tk.Frame(root, bg="#1e293b", height=120)
top_frame.pack(fill="x")

title_label = tk.Label(
    top_frame,
    text="🖐 Hand Gesture Application",
    bg="#1e293b",
    fg="white",
    font=title_font
)
title_label.pack(pady=(25, 5))

subtitle_label = tk.Label(
    top_frame,
    text="Touchless • Intelligent • Real-time Interaction",
    bg="#1e293b",
    fg="#93c5fd",
    font=subtitle_font
)
subtitle_label.pack()

# ------------------ CENTER FRAME ------------------
center_frame = tk.Frame(root, bg="#0f172a")
center_frame.pack(expand=True)

status_label = tk.Label(
    center_frame,
    text="Select a service to begin",
    bg="#0f172a",
    fg="#a5b4fc",
    font=status_font
)
status_label.pack(pady=10)

# ------------------ BUTTON ACTIONS ------------------
def start_gesture_prediction():
    status_label.config(text="🎥 Starting Gesture Prediction...")
    threading.Thread(target=real_time_detection.run, daemon=True).start()

def coming_soon(feature):
    status_label.config(text=f"⚙️ {feature} coming soon...")

# ------------------ BUTTON STYLE ------------------
def create_button(text, command, bg_color):
    return tk.Button(
        center_frame,
        text=text,
        command=command,
        bg=bg_color,
        fg="white",
        activebackground="#334155",
        activeforeground="white",
        font=btn_font,
        width=30,
        height=2,
        bd=0,
        cursor="hand2"
    )

# ------------------ BUTTONS ------------------
btn_gesture = create_button(
    "🎥 Gesture Prediction",
    start_gesture_prediction,
    "#2563eb"
)
btn_gesture.pack(pady=12)

btn_control = create_button(
    "🎮 Gesture Control",
    lambda: coming_soon("Gesture Control"),
    "#16a34a"
)
btn_control.pack(pady=12)

btn_sign = create_button(
    "🤟 Sign Language Mode",
    lambda: coming_soon("Sign Language Mode"),
    "#9333ea"
)
btn_sign.pack(pady=12)

btn_exit = create_button(
    "❌ Exit Application",
    root.quit,
    "#dc2626"
)
btn_exit.pack(pady=(25, 10))

# ------------------ FOOTER ------------------
footer = tk.Label(
    root,
    text="Developed as an AI-based Human–Computer Interaction System",
    bg="#0f172a",
    fg="#64748b",
    font=("Segoe UI", 9)
)
footer.pack(pady=8)

root.mainloop()
