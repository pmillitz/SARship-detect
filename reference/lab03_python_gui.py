import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import scipy.ndimage as ndi

class ImageProcessingGUI:
    def __init__(self, master):
        self.master = master
        master.title("Image Processing GUI")
        master.geometry("800x500")
        
        # Initialize variables
        self.original_image = None
        self.processed_image = None
        
        # Create frames for layout
        self.image_frame = tk.Frame(master)
        self.image_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        self.control_frame = tk.Frame(master)
        self.control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        # Create canvas for displaying images
        self.original_canvas = tk.Canvas(self.image_frame, width=350, height=350, bg="gray")
        self.original_canvas.grid(row=0, column=0, padx=5, pady=5)
        
        self.processed_canvas = tk.Canvas(self.image_frame, width=350, height=350, bg="gray")
        self.processed_canvas.grid(row=0, column=1, padx=5, pady=5)
        
        # Create buttons
        self.load_button = tk.Button(self.control_frame, text="Load Image", width=12, command=self.load_image)
        self.load_button.grid(row=0, column=0, padx=10, pady=5)
        
        self.hist_eq_button = tk.Button(self.control_frame, text="Histogram Equalize", width=15, command=self.histogram_equalize)
        self.hist_eq_button.grid(row=0, column=1, padx=10, pady=5)
        
        self.high_boost_button = tk.Button(self.control_frame, text="High Boost", width=12, command=self.high_boost)
        self.high_boost_button.grid(row=0, column=2, padx=10, pady=5)
        
        # Boost factor input
        self.boost_factor_var = tk.StringVar(value="10")
        self.boost_factor_entry = tk.Entry(self.control_frame, textvariable=self.boost_factor_var, width=5)
        self.boost_factor_entry.grid(row=0, column=3, pady=5)
        
        # Low pass filter controls
        self.low_pass_label = tk.Label(self.control_frame, text="Low Pass")
        self.low_pass_label.grid(row=1, column=0, padx=10, pady=5)
        
        self.filter_size_var = tk.StringVar(value="3")
        self.filter_size_entry = tk.Entry(self.control_frame, textvariable=self.filter_size_var, width=5)
        self.filter_size_entry.grid(row=1, column=1, pady=5)
        
        self.sigma_var = tk.StringVar(value="2")
        self.sigma_entry = tk.Entry(self.control_frame, textvariable=self.sigma_var, width=5)
        self.sigma_entry.grid(row=1, column=2, pady=5)
        
        # High pass filter button
        self.high_pass_button = tk.Button(self.control_frame, text="High Pass", width=12, command=self.high_pass)
        self.high_pass_button.grid(row=2, column=1, padx=10, pady=5)
        
        # Low pass filter button
        self.low_pass_button = tk.Button(self.control_frame, text="Low Pass", width=12, command=self.low_pass)
        self.low_pass_button.grid(row=2, column=0, padx=10, pady=5)
        
        # Keep references to the images to prevent garbage collection
        self.original_photo = None
        self.processed_photo = None
    
    def load_image(self):
        """Load an image and display it on the left canvas"""
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp *.tif *.tiff")]
        )
        
        if file_path:
            # Read the image using OpenCV
            self.original_image = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            
            if self.original_image is None:
                messagebox.showerror("Error", "Could not load image")
                return
            
            # Display the original image
            self.display_image(self.original_image, self.original_canvas, "original")
    
    def display_image(self, image, canvas, img_type):
        """Display an image on the specified canvas"""
        # Resize image to fit canvas if necessary
        canvas_width = canvas.winfo_width()
        canvas_height = canvas.winfo_height()
        
        # Ensure dimensions are valid
        if canvas_width <= 1:
            canvas_width = 350
        if canvas_height <= 1:
            canvas_height = 350
            
        # Calculate resize ratio
        height, width = image.shape[:2]
        ratio = min(canvas_width/width, canvas_height/height)
        new_size = (int(width * ratio), int(height * ratio))
        
        # Resize image
        resized_image = cv2.resize(image, new_size)
        
        # Convert to PIL format for tkinter
        pil_image = Image.fromarray(resized_image)
        
        # Convert to PhotoImage format
        if img_type == "original":
            self.original_photo = ImageTk.PhotoImage(image=pil_image)
            canvas.create_image(canvas_width//2, canvas_height//2, image=self.original_photo, anchor=tk.CENTER)
        else:
            self.processed_photo = ImageTk.PhotoImage(image=pil_image)
            canvas.create_image(canvas_width//2, canvas_height//2, image=self.processed_photo, anchor=tk.CENTER)
    
    def check_image_loaded(self):
        """Check if an image is loaded and show warning if not"""
        if self.original_image is None:
            messagebox.showwarning("Warning", "Please load an image first")
            return False
        return True
    
    def histogram_equalize(self):
        """Apply histogram equalization to the image"""
        if not self.check_image_loaded():
            return
        
        # Apply histogram equalization using OpenCV
        self.processed_image = cv2.equalizeHist(self.original_image)
        
        # Display the processed image
        self.display_image(self.processed_image, self.processed_canvas, "processed")
    
    def low_pass(self):
        """Apply a Gaussian low-pass filter to the image"""
        if not self.check_image_loaded():
            return
        
        try:
            # Get filter parameters
            filter_size = int(self.filter_size_var.get())
            sigma = float(self.sigma_var.get())
            
            # Ensure filter size is valid
            if filter_size < 1:
                raise ValueError("Filter size must be positive")
            
            # Ensure filter size is odd
            if filter_size % 2 == 0:
                filter_size += 1
                self.filter_size_var.set(str(filter_size))
            
            # Apply Gaussian filter
            self.processed_image = cv2.GaussianBlur(
                self.original_image, 
                (filter_size, filter_size), 
                sigma
            )
            
            # Display the processed image
            self.display_image(self.processed_image, self.processed_canvas, "processed")
            
        except (ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Invalid filter parameters: {str(e)}")
    
    def high_pass(self):
        """Apply a high-pass filter to the image"""
        if not self.check_image_loaded():
            return
        
        # Define Laplacian kernel for high-pass filtering
        kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
        
        # Apply the filter
        self.processed_image = cv2.filter2D(self.original_image, -1, kernel)
        
        # Normalize to display range
        self.processed_image = cv2.normalize(self.processed_image, None, 0, 255, cv2.NORM_MINMAX)
        
        # Display the processed image
        self.display_image(self.processed_image, self.processed_canvas, "processed")
    
    def high_boost(self):
        """Apply a high-boost filter to the image"""
        if not self.check_image_loaded():
            return
        
        try:
            # Get boost factor
            boost_factor = float(self.boost_factor_var.get())
            
            # Create a blurred version using a 3x3 Gaussian
            blurred = cv2.GaussianBlur(self.original_image, (3, 3), 0.5)
            
            # Calculate high-pass (original - blurred)
            high_pass = self.original_image.astype(float) - blurred.astype(float)
            
            # Apply high boost formula: original + k*(high-pass)
            boosted = self.original_image.astype(float) + boost_factor * high_pass
            
            # Clip values to valid range
            self.processed_image = np.clip(boosted, 0, 255).astype(np.uint8)
            
            # Display the processed image
            self.display_image(self.processed_image, self.processed_canvas, "processed")
            
        except (ValueError, TypeError) as e:
            messagebox.showerror("Error", f"Invalid boost factor: {str(e)}")

# Create the main window
if __name__ == "__main__":
    root = tk.Tk()
    app = ImageProcessingGUI(root)
    root.mainloop()
