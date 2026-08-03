import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
from PIL import Image, ImageTk
import tensorflow as tf

try:
    from tkinterdnd2 import DND_FILES, TkinterDnD
    DND_AVAILABLE = True
except ImportError:
    DND_AVAILABLE = False

MODELS_DIR = "." 

CLASSES = ["buildings", "forest", "glacier", "mountain", "sea", "street"]

MODELS = {
    "VGG16 (fine-tuning) — best performance": {
        "file": "vgg16_finetuning_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "VGG16 (scratch)": {
        "file": "vgg16_scratch_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "VGG16 (feature extraction)": {
        "file": "vgg16_feature_extraction_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "VGG16 (transfer learning)": {
        "file": "vgg16_tl_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.vgg16.preprocess_input,
    },
    "ResNet50 (fine-tuning)": {
        "file": "resnet50_finetuning_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
    },
    "ResNet50 (scratch)": {
        "file": "resnet50_scratch_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
    },
    "ResNet50 (transfer learning)": {
        "file": "resnet50_tl_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.resnet50.preprocess_input,
    },
    "DenseNet121 (fine-tuning)": {
        "file": "densenet121_finetuning_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet121 (scratch)": {
        "file": "densenet121_scratch_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "DenseNet121 (transfer learning)": {
        "file": "densenet121_tl_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.densenet.preprocess_input,
    },
    "InceptionV3 (fine-tuning)": {
        "file": "inceptionv3_finetuning_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
    "InceptionV3 (scratch)": {
        "file": "inceptionv3_scratch_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
    "InceptionV3 (transfer learning)": {
        "file": "inceptionv3_tl_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.inception_v3.preprocess_input,
    },
    "MobileNetV2 (fine-tuning)": {
        "file": "mobilenetv2_finetuning_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "MobileNetV2 (scratch)": {
        "file": "mobilenetv2_scratch_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "MobileNetV2 (transfer learning)": {
        "file": "mobilenetv2_tl_best.keras",
        "input_size": (150, 150),
        "preprocess": tf.keras.applications.mobilenet_v2.preprocess_input,
    },
    "Custom CNN (scratch)": {
        "file": "custom_cnn_best.keras",
        "input_size": (150, 150),
        "preprocess": lambda x: x / 255.0,
    },
}

AVAILABLE_MODELS = {
    name: cfg for name, cfg in MODELS.items()
    if os.path.isfile(os.path.join(MODELS_DIR, cfg["file"]))
}

BG = "#ffffff"
BG_MUTED = "#fafafa"
FG = "#000000"
FG_MUTED = "#666666"
BORDER = "#eaeaea"
BORDER_HOVER = "#000000"
FONT = "TkDefaultFont"

BaseClass = TkinterDnD.Tk if DND_AVAILABLE else tk.Tk


class App(BaseClass):
    def __init__(self):
        super().__init__()
        self.title("Natural Scene Classifier")
        self.geometry("460x680")
        self.configure(bg=BG)
        self.resizable(False, False)

        self.loaded_models = {}
        self.pil_image = None

        if not AVAILABLE_MODELS:
            messagebox.showerror(
                "No models",
                f"No .keras file was found in '{MODELS_DIR}'.\n"
                "Adjust MODELS_DIR at the top of the script.",
            )
            self.destroy()
            return

        self._setup_style()
        self._build_ui()

    # ---------- style ----------
    def _setup_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")

        style.configure("TFrame", background=BG)
        style.configure("TLabel", background=BG, foreground=FG, font=(FONT, 10))
        style.configure("Muted.TLabel", background=BG, foreground=FG_MUTED, font=(FONT, 9))
        style.configure("Title.TLabel", background=BG, foreground=FG, font=(FONT, 17, "bold"))
        style.configure("Field.TLabel", background=BG, foreground=FG_MUTED, font=(FONT, 8, "bold"))

        style.configure(
            "TCombobox",
            fieldbackground=BG, background=BG, foreground=FG, arrowcolor=FG,
            bordercolor=BORDER, lightcolor=BORDER, darkcolor=BORDER,
            padding=8, relief="flat", arrowsize=12,
        )
        style.map("TCombobox", fieldbackground=[("readonly", BG)])

        style.configure(
            "Accent.TButton",
            background=FG, foreground=BG, borderwidth=0,
            focuscolor=FG, padding=(16, 11), font=(FONT, 10, "bold"),
        )
        style.map(
            "Accent.TButton",
            background=[("active", "#333333"), ("disabled", "#d4d4d4")],
            foreground=[("disabled", "#888888")],
        )

    # ---------- UI construction ----------
    def _build_ui(self):
        container = ttk.Frame(self, padding=28)
        container.pack(fill="both", expand=True)

        ttk.Label(container, text="Scene Classifier", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            container, text="Choose a model and upload or drag an image.",
            style="Muted.TLabel",
        ).pack(anchor="w", pady=(4, 20))

        ttk.Label(container, text="MODEL", style="Field.TLabel").pack(anchor="w")
        self.model_combo = ttk.Combobox(
            container, values=list(AVAILABLE_MODELS.keys()), state="readonly"
        )
        self.model_combo.current(0)
        self.model_combo.pack(fill="x", pady=(6, 20))

        # --- Dropzone ---
        self.dropzone = tk.Frame(
            container, bg=BG_MUTED, height=220,
            highlightbackground=BORDER, highlightthickness=1,
        )
        self.dropzone.pack(fill="x")
        self.dropzone.pack_propagate(False)

        self.image_label = tk.Label(
            self.dropzone, text=self._dropzone_text(),
            bg=BG_MUTED, fg=FG_MUTED, font=(FONT, 9),
            justify="center", cursor="hand2",
        )
        self.image_label.pack(expand=True, fill="both")

        for widget in (self.dropzone, self.image_label):
            widget.bind("<Button-1>", lambda e: self.upload_image())
            widget.bind("<Enter>", self._hover_dropzone_on)
            widget.bind("<Leave>", self._hover_dropzone_off)

        if DND_AVAILABLE:
            self.dropzone.drop_target_register(DND_FILES)
            self.dropzone.dnd_bind("<<Drop>>", self._on_drop)
            self.image_label.drop_target_register(DND_FILES)
            self.image_label.dnd_bind("<<Drop>>", self._on_drop)

        # --- Classify button ---
        self.classify_button = ttk.Button(
            container, text="Classify image", style="Accent.TButton",
            command=self.classify, state="disabled",
        )
        self.classify_button.pack(fill="x", pady=(16, 0))

        self.status_label = ttk.Label(container, text="", style="Muted.TLabel")
        self.status_label.pack(anchor="w", pady=(10, 0))

        self.results_frame = ttk.Frame(container)
        self.results_frame.pack(fill="x", pady=(12, 0))

    def _dropzone_text(self):
        if DND_AVAILABLE:
            return "Drag an image here\n— or —\nclick to upload"
        return "Click to upload an image\n\n(install 'tkinterdnd2' for drag and drop)"

    def _hover_dropzone_on(self, event):
        self.dropzone.configure(highlightbackground=BORDER_HOVER)

    def _hover_dropzone_off(self, event):
        self.dropzone.configure(highlightbackground=BORDER)

    def _on_drop(self, event):
        path = event.data.strip("{}")
        if path.lower().endswith((".jpg", ".jpeg", ".png")):
            self._load_image(path)
        else:
            messagebox.showwarning("Invalid format", "Only .jpg, .jpeg or .png images are accepted")

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.jpeg *.png")])
        if path:
            self._load_image(path)

    def _load_image(self, path):
        self.pil_image = Image.open(path).convert("RGB")

        preview = self.pil_image.copy()
        preview.thumbnail((260, 200))
        self.tk_image = ImageTk.PhotoImage(preview)
        self.image_label.configure(image=self.tk_image, text="", bg=BG_MUTED)
        self.classify_button.configure(state="normal")

        for widget in self.results_frame.winfo_children():
            widget.destroy()
        self.status_label.configure(text="")

    # ---------- classification ----------
    def classify(self):
        if self.pil_image is None:
            return

        model_name = self.model_combo.get()
        config = AVAILABLE_MODELS[model_name]

        self.status_label.configure(text="Loading model...")
        self.update_idletasks()

        if model_name not in self.loaded_models:
            path = os.path.join(MODELS_DIR, config["file"])
            self.loaded_models[model_name] = tf.keras.models.load_model(path)
        model = self.loaded_models[model_name]

        self.status_label.configure(text="Classifying...")
        self.update_idletasks()

        resized_img = self.pil_image.resize(config["input_size"])
        arr = np.expand_dims(np.array(resized_img, dtype=np.float32), axis=0)
        prepped_arr = config["preprocess"](arr)
        pred = model.predict(prepped_arr, verbose=0)[0]

        idx_top = np.argsort(pred)[::-1][:3]

        for widget in self.results_frame.winfo_children():
            widget.destroy()

        ttk.Label(self.results_frame, text="RESULTS", style="Field.TLabel").pack(anchor="w", pady=(0, 8))
        for i in idx_top:
            label = CLASSES[i] if i < len(CLASSES) else f"Class {i}"
            self._add_result_row(label, float(pred[i]))

        self.status_label.configure(text="Done.")

    def _add_result_row(self, label, value):
        row = ttk.Frame(self.results_frame)
        row.pack(fill="x", pady=4)

        header = ttk.Frame(row)
        header.pack(fill="x")
        ttk.Label(header, text=label, font=(FONT, 10, "bold")).pack(side="left")
        ttk.Label(header, text=f"{value * 100:.1f}%", style="Muted.TLabel").pack(side="right")

        track = tk.Canvas(row, height=6, bg="#f0f0f0", highlightthickness=0)
        track.pack(fill="x", pady=(6, 0))
        track.update_idletasks()
        track.bind(
            "<Configure>",
            lambda e, c=track, v=value: (c.delete("bar"), c.create_rectangle(
                0, 0, e.width * v, e.height, fill=FG, width=0, tags="bar"
            )),
        )


if __name__ == "__main__":
    app = App()
    app.mainloop()
