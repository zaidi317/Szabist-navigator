"""
Campus Location Identifier GUI using Tkinter
Simple interface to upload an image and identify its location
"""

import logging
import sys
from pathlib import Path
from tkinter import Tk, Frame, Label, Button, filedialog, messagebox, StringVar
from tkinter import ttk
from PIL import Image, ImageTk
import threading

# Setup paths
BASE_DIR = Path(__file__).parent.absolute()
sys.path.insert(0, str(BASE_DIR))

from predict_finetuned import FineTunedLocationPredictor

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CampusLocationGUI:
    """Tkinter GUI for campus location identification"""

    def __init__(self, root: Tk):
        self.root = root
        self.root.title("Campus Location Identifier")
        self.root.geometry("900x700")
        self.root.resizable(True, True)

        self.model = None
        self.current_image_path = None
        self.checkpoint_path = BASE_DIR / "checkpoints" / "best_finetuned_dinov2.pth"

        self._setup_ui()
        self._load_model()

    def _setup_ui(self) -> None:
        """Build the GUI layout"""
        # Title
        title_label = Label(
            self.root,
            text="Campus Location Identifier",
            font=("Arial", 18, "bold"),
        )
        title_label.pack(pady=10)

        # Main container
        main_frame = Frame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left side: Image display
        left_frame = Frame(main_frame)
        left_frame.pack(side="left", fill="both", expand=True, padx=5)

        Label(left_frame, text="Image Preview:", font=("Arial", 12, "bold")).pack(
            anchor="w"
        )

        self.image_label = Label(
            left_frame, bg="lightgray", width=40, height=25, text="No image loaded"
        )
        self.image_label.pack(fill="both", expand=True, pady=5)

        # Buttons
        button_frame = Frame(left_frame)
        button_frame.pack(fill="x", pady=5)

        self.upload_btn = Button(
            button_frame,
            text="📁 Upload Image",
            font=("Arial", 10, "bold"),
            command=self.upload_image,
            bg="#4CAF50",
            fg="white",
            padx=10,
            pady=8,
        )
        self.upload_btn.pack(side="left", padx=5)

        self.predict_btn = Button(
            button_frame,
            text="🔍 Identify Location",
            font=("Arial", 10, "bold"),
            command=self.predict_location,
            bg="#2196F3",
            fg="white",
            padx=10,
            pady=8,
            state="disabled",
        )
        self.predict_btn.pack(side="left", padx=5)

        self.clear_btn = Button(
            button_frame,
            text="🗑️ Clear",
            font=("Arial", 10, "bold"),
            command=self.clear_image,
            bg="#FF9800",
            fg="white",
            padx=10,
            pady=8,
        )
        self.clear_btn.pack(side="left", padx=5)

        # Right side: Results
        right_frame = Frame(main_frame)
        right_frame.pack(side="right", fill="both", expand=True, padx=5)

        Label(right_frame, text="Results:", font=("Arial", 12, "bold")).pack(
            anchor="w"
        )

        # Results text area with scrollbar
        result_frame = Frame(right_frame)
        result_frame.pack(fill="both", expand=True, pady=5)

        scrollbar = ttk.Scrollbar(result_frame)
        scrollbar.pack(side="right", fill="y")

        self.result_text = ttk.Label(
            result_frame,
            text="Results will appear here...",
            font=("Arial", 10),
            background="white",
            foreground="black",
            justify="left",
            wraplength=350,
        )
        self.result_text.pack(fill="both", expand=True, anchor="nw")

        # Status bar
        self.status_var = StringVar(value="Ready")
        status_bar = ttk.Label(
            self.root, textvariable=self.status_var, relief="sunken", anchor="w"
        )
        status_bar.pack(fill="x", side="bottom", padx=5, pady=5)

    def _load_model(self) -> None:
        """Load the fine-tuned predictor in background thread"""

        def load():
            try:
                self.status_var.set("Loading fine-tuned model... Please wait")
                self.root.update()
                if not self.checkpoint_path.exists():
                    raise FileNotFoundError(
                        f"Fine-tuned checkpoint not found: {self.checkpoint_path}. "
                        "Run finetune_dinov2.py first."
                    )

                logger.info("Loading FineTunedLocationPredictor model...")
                self.model = FineTunedLocationPredictor(self.checkpoint_path)
                logger.info("Fine-tuned model loaded successfully!")
                self.status_var.set("Ready")
            except Exception as e:
                logger.error(f"Failed to load model: {e}")
                self.status_var.set(f"Error: {e}")
                messagebox.showerror("Error", f"Failed to load model:\n{e}")

        thread = threading.Thread(target=load, daemon=True)
        thread.start()

    def upload_image(self) -> None:
        """Open file dialog to select an image"""
        file_types = [
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.tiff"),
            ("All files", "*.*"),
        ]
        file_path = filedialog.askopenfilename(filetypes=file_types)

        if file_path:
            self.current_image_path = Path(file_path)
            self.display_image(self.current_image_path)
            self.predict_btn.config(state="normal")
            self.result_text.config(text="Image loaded. Click 'Identify Location' to predict.")
            self.status_var.set(f"Loaded: {self.current_image_path.name}")

    def display_image(self, image_path: Path) -> None:
        """Display image in the preview label"""
        try:
            img = Image.open(image_path)
            # Resize to fit label (400x350)
            img.thumbnail((400, 350), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(img)

            self.image_label.config(image=photo, text="")
            self.image_label.image = photo  # Keep a reference
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image:\n{e}")

    def predict_location(self) -> None:
        """Predict location and display results"""
        if not self.current_image_path or not self.model:
            messagebox.showwarning("Warning", "Please load an image and wait for model to load.")
            return

        # Run prediction in background thread to avoid freezing UI
        def predict():
            try:
                self.status_var.set("Predicting location...")
                self.root.update()

                result = self.model.predict(self.current_image_path)

                if "error" in result:
                    self.display_error(result["error"])
                else:
                    self.display_results(result)
                    self.status_var.set("Prediction complete")
            except Exception as e:
                logger.error(f"Prediction error: {e}")
                self.display_error(str(e))
                self.status_var.set("Error during prediction")

        thread = threading.Thread(target=predict, daemon=True)
        thread.start()

    def display_results(self, result: dict) -> None:
        """Display prediction results in results panel"""
        predicted = result.get("predicted_location", "Unknown")
        confidence = result.get("confidence", 0)
        top_predictions = result.get("top_predictions", [])

        # Format results text
        results_text = f"""
PREDICTION RESULT
{'='*40}

📍 LOCATION: {predicted}

📊 CONFIDENCE: {confidence * 100:.1f}%

🏆 TOP PREDICTIONS:
"""
        for pred in top_predictions:
            label = pred.get("label", "Unknown")
            probability = pred.get("probability", 0)
            results_text += f"   • {label}: {probability * 100:.1f}%\n"

        results_text += f"\n🔍 RANKED RESULTS:\n"
        for i, pred in enumerate(top_predictions, 1):
            label = pred.get("label", "Unknown")
            probability = pred.get("probability", 0)
            results_text += f"   {i}. {label} ({probability * 100:.2f}%)\n"

        self.result_text.config(text=results_text)

    def display_error(self, error_msg: str) -> None:
        """Display error message"""
        self.result_text.config(text=f"❌ ERROR:\n\n{error_msg}")
        messagebox.showerror("Error", f"Prediction failed:\n{error_msg}")

    def clear_image(self) -> None:
        """Clear the current image and results"""
        self.current_image_path = None
        self.image_label.config(image="", text="No image loaded")
        self.image_label.image = None
        self.result_text.config(text="Results will appear here...")
        self.predict_btn.config(state="disabled")
        self.status_var.set("Ready")


def main() -> None:
    root = Tk()
    app = CampusLocationGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
