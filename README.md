# TN_Seg - Trigeminal Nerve Segmentation Extension for 3D Slicer

Automated segmentation of trigeminal nerve and vessels using ONNX deep learning model.

## Requirements

- 3D Slicer 5.4 or later
- Python packages:
  - numpy
  - scipy
  - onnxruntime

## Installation

### Method 1: Install from ZIP (Recommended for Sharing)

1. **For the extension developer:**
   - In 3D Slicer, go to `File` → `Add Extension`
   - Select the `TNSeg-v1.0.zip` file
   - Restart Slicer

2. **For end users:**
   - Extract the `TN_auto_seg` folder to any location
   - In 3D Slicer, go to `File` → `Add Extension` → `Browse`
   - Select the extracted `TN_auto_seg` folder
   - Restart Slicer

### Method 2: Manual Installation

1. Copy the entire `TN_auto_seg` folder to:
   ```
   C:\Users\<YourUsername>\AppData\Local\NA-MIC\Slicer 5.8\Extensions\
   ```
   (Create the folder if it doesn't exist)

2. Copy the ONNX model file (`trigeminal_3d_fullres.onnx`) to the same directory as `TN_Seg.py`

3. Restart 3D Slicer

## Usage

1. **Load MRI Data:**
   - Drag and drop your MRI volume into Slicer
   - Or use `File` → `Add Data`

2. **Run Segmentation:**
   - Go to: `Modules` → `Segmentation` → `TN_Seg`
   - Select input MRI volume from the dropdown
   - Choose or create an output LabelMap
   - Click `Run Segmentation`

3. **View Results:**
   - The segmentation will be displayed as a LabelMap in the segment editor
   - Colors:
     - Red: Right Trigeminal Nerve (R_Trigeminal)
     - Green: Left Trigeminal Nerve (L_Trigeminal)
     - Blue: Blood Vessel (Vessel)

## Model Information

- **Architecture:** 3D U-Net based (nnU-Net)
- **Input Size:** 40 × 224 × 256
- **Classes:**
  - Background (0)
  - Right Trigeminal Nerve (1)
  - Left Trigeminal Nerve (2)
  - Blood Vessel (3)

## Troubleshooting

### "onnxruntime not installed"
Run in Slicer's Python console:
```
import subprocess
subprocess.run(["pip", "install", "onnxruntime"])
```

### Segmentation appears rotated or misaligned
The model output is optimized for the input MRI space. Make sure your MRI data is in the correct orientation before running segmentation.

### Out of memory errors
For large MRI volumes, the extension uses sliding window inference. This is normal and may take longer.

## File Structure

```
TN_auto_seg/
├── CMakeLists.txt
├── TN_auto_seg.png
├── trigeminal_3d_fullres.onnx    # ONNX model (116 MB)
└── TN_Seg/
    ├── CMakeLists.txt
    ├── TN_Seg.py                  # Main Python module
    └── Resources/
        ├── Icons/TN_Seg.png
        └── UI/TN_Seg.ui
```

## Citation

If you use this extension in your research, please cite:
> [Your citation here]

## License

MIT License

## Contact

For issues or questions, please contact: [Your email]