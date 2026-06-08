import logging
import os
import numpy as np
from scipy.special import softmax
import random

import vtk
import qt
import slicer
from slicer.i18n import tr as _
from slicer.ScriptedLoadableModule import *


# ============================================================
# Configuration
# ============================================================
# Extension directory path
EXTENSION_DIR = os.path.dirname(os.path.dirname(__file__))
MODEL_PATH = os.path.join(EXTENSION_DIR, "trigeminal_3d_fullres.onnx")

# Patch size (must match training configuration)
PATCH_D = 40
PATCH_H = 224
PATCH_W = 256

# Number of classes
NUM_CLASSES = 4
CLASS_NAMES = {0: 'Background', 1: 'R_Trigeminal', 2: 'L_Trigeminal', 3: 'Vessel'}


# ============================================================
# TN_Seg
# ============================================================

class TN_Seg(ScriptedLoadableModule):
    """Trigeminal Nerve Segmentation - Automated segmentation of trigeminal nerve and vessels using ONNX model"""

    def __init__(self, parent):
        ScriptedLoadableModule.__init__(self, parent)
        self.parent.title = _("TN_Seg")
        self.parent.categories = ["Segmentation"]
        self.parent.dependencies = []
        self.parent.contributors = ["Your Name/Institution"]  # TODO: Replace with actual contributors
        self.parent.helpText = _("Trigeminal Nerve Segmentation - Automated segmentation of trigeminal nerve and vessels using ONNX model")
        self.parent.acknowledgementText = _("Automated segmentation using deep learning")


# ============================================================
# TN_SegWidget
# ============================================================

class TN_SegWidget(ScriptedLoadableModuleWidget):
    """Trigeminal Nerve Segmentation UI"""

    def __init__(self, parent=None):
        ScriptedLoadableModuleWidget.__init__(self, parent)
        self.logic = None

    def setup(self):
        ScriptedLoadableModuleWidget.setup(self)

        # Load UI file
        self.ui = slicer.util.loadUI(self.resourcePath("UI/TN_Seg.ui"))
        self.layout.addWidget(self.ui)

        # Setup UI references
        self.inputSelector = self.ui.findChild(slicer.qMRMLNodeComboBox, "inputSelector")
        self.outputSelector = self.ui.findChild(slicer.qMRMLNodeComboBox, "outputSelector")
        self.applyButton = self.ui.findChild(qt.QPushButton, "applyButton")

        # Connect MRML scene
        self.inputSelector.setMRMLScene(slicer.mrmlScene)
        self.outputSelector.setMRMLScene(slicer.mrmlScene)

        # Check onnxruntime
        try:
            import onnxruntime as ort
        except ImportError:
            logging.error("onnxruntime not installed. Please run: pip install onnxruntime")
            self.applyButton.setEnabled(False)
            return

        # Initialize logic
        self.logic = TN_SegLogic()


        # Check model file
        if os.path.exists(MODEL_PATH):
            logging.info(f"Model loaded: {os.path.basename(MODEL_PATH)}")
        else:
            logging.warning(f"Model file not found: {MODEL_PATH}")
            self.applyButton.setEnabled(False)

        # Connect signals
        self.applyButton.clicked.connect(self.onApply)
        self.inputSelector.currentNodeChanged.connect(self.onInputChanged)

        # Initial state
        self.onInputChanged(self.inputSelector.currentNode())

    def onInputChanged(self, node):
        """Input node changed callback"""
        self.applyButton.setEnabled(node is not None)

    def onApply(self):
        """Run inference"""
        inputNode = self.inputSelector.currentNode()
        outputNode = self.outputSelector.currentNode()

        if not inputNode:
            return

        # Create output node
        if not outputNode:
            outputNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
            outputNode.SetName("TN_LabelMap")
            self.outputSelector.setCurrentNode(outputNode)

        # Disable button
        self.applyButton.setEnabled(False)

        try:
            # Run inference
            result = self.logic.runInference(inputNode)
            
            # Create labelmap and segmentation node
            segNode = self.logic.createSegmentation(result, inputNode, outputNode)
            
            # Set segmentation as active (similar to load_predictions_for_slicer.py)
            if segNode:
                slicer.app.applicationLogic().GetSelectionNode().SetReferenceActiveSegmentID(segNode.GetID())
                slicer.app.applicationLogic().PropagateVolumeSelection(0)
            
            logging.info("Segmentation completed!")
            
        except Exception as e:
            logging.error(f"Error: {str(e)}")
            import traceback
            traceback.print_exc()

        # Enable button
        self.applyButton.setEnabled(True)


# ============================================================
# TN_SegLogic
# ============================================================

class TN_SegLogic(ScriptedLoadableModuleLogic):
    """ONNX inference logic"""

    def __init__(self):
        ScriptedLoadableModuleLogic.__init__(self)
        self.session = None
        self.last_rotation = None  # Store last rotation info (axis, k)
        self.inputName = None
        self.outputName = None
        self._loadModel()

    def _loadModel(self):
        """Load ONNX model"""
        try:
            import onnxruntime as ort
        except ImportError:
            raise ImportError("Please install onnxruntime: pip install onnxruntime")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

        logging.info(f"Loading ONNX model: {MODEL_PATH}")
        self.session = ort.InferenceSession(
            MODEL_PATH,
            providers=['CPUExecutionProvider']
        )
        self.inputName = self.session.get_inputs()[0].name
        self.outputName = self.session.get_outputs()[0].name
        logging.info(f"Input: {self.inputName}, Output: {self.outputName}")

    def createCoordinateChannels(self, D, H, W):
        """Create 3D coordinate channels"""
        coord_x = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, 1, 1, 1, W)
        coord_x = np.broadcast_to(coord_x, (1, 1, D, H, W)).astype(np.float32)

        coord_y = np.linspace(-1, 1, H, dtype=np.float32).reshape(1, 1, 1, H, 1)
        coord_y = np.broadcast_to(coord_y, (1, 1, D, H, W)).astype(np.float32)

        coord_z = np.linspace(-1, 1, D, dtype=np.float32).reshape(1, 1, D, 1, 1)
        coord_z = np.broadcast_to(coord_z, (1, 1, D, H, W)).astype(np.float32)

        return coord_x, coord_y, coord_z

    def preprocess(self, mri_array):
        """Z-Score normalization"""
        mri = mri_array.astype(np.float32)
        mean = np.mean(mri)
        std = np.std(mri)
        if std > 1e-8:
            mri = (mri - mean) / std
        else:
            mri = mri - mean
        return mri

    def runInference(self, inputVolumeNode, debug_rotate=None):
        """Run inference
        
        Args:
            inputVolumeNode: Input MRI volume node
            debug_rotate: Debug mode, randomly rotate the output labelmap.
                        True=random rotate, or specify (axis, k) tuple
        """
        # Get MRI data
        mriArray = slicer.util.arrayFromVolume(inputVolumeNode)
        D, H, W = mriArray.shape

        logging.info(f"Input volume: {D}x{H}x{W}")

        # Preprocess
        mriNorm = self.preprocess(mriArray)

        # Check if sliding window is needed
        if D <= PATCH_D and H <= PATCH_H and W <= PATCH_W:
            # Direct inference
            mri_ch = mriNorm[np.newaxis, np.newaxis].astype(np.float32)
            cx, cy, cz = self.createCoordinateChannels(D, H, W)
            inp = np.concatenate([mri_ch, cx, cy, cz], axis=1)

            logits = self.session.run([self.outputName], {self.inputName: inp})[0][0]
            prob = softmax(logits, axis=0)
            seg = np.argmax(prob, axis=0)
        else:
            # Sliding window inference
            seg = self._slidingWindowInference(mriNorm)
        
        # Debug mode: randomly rotate the output labelmap
        rotation_info = None
        if debug_rotate is not None:
            if debug_rotate is True:
                # Randomly select axis and k
                axes = ['D', 'H', 'W']
                axis = random.choice(axes)
                k = random.choice([1, 2, 3])  # 1=90°, 2=180°, 3=270°
                debug_rotate = (axis, k)
            
            if isinstance(debug_rotate, tuple) and len(debug_rotate) == 2:
                axis, k = debug_rotate
                seg = self._rotate_array(seg, axis, k)
                self.last_rotation = (axis, k)  # Store rotation info
                rotation_info = f"debug: rotated output along {axis} axis by {k*90} degrees"
                logging.info(f"Debug mode: {rotation_info}")
        
        return seg.astype(np.uint8)
    
    def _rotate_array(self, arr, axis, k):
        """Rotate 3D array"""
        if k == 0:
            return arr
        
        axis_map = {'D': 0, 'H': 1, 'W': 2}
        ax = axis_map.get(axis, 0)
        
        return np.rot90(arr, k=-k, axes=(ax, (ax+1)%3))

    def _slidingWindowInference(self, mriNorm):
        """Sliding window inference"""
        D, H, W = mriNorm.shape
        pD, pH, pW = PATCH_D, PATCH_H, PATCH_W
        sD, sH, sW = max(1, int(pD * 0.5)), max(1, int(pH * 0.5)), max(1, int(pW * 0.5))

        # Generate positions
        positions = []
        for d in range(0, D - pD + 1, sD):
            for h in range(0, H - pH + 1, sH):
                for w in range(0, W - pW + 1, sW):
                    positions.append((d, h, w))

        # Boundary handling
        if D > pD and (D - pD) % sD != 0:
            positions.append((D - pD, 0, 0))
        if H > pH and (H - pH) % sH != 0:
            for d in range(0, D - pD + 1, sD):
                positions.append((d, H - pH, 0))
        if W > pW and (W - pW) % sW != 0:
            for d in range(0, D - pD + 1, sD):
                for h in range(0, H - pH + 1, sD):
                    positions.append((d, h, W - pW))

        positions = list(set(positions))
        logging.info(f"Sliding window: {len(positions)} patches")

        # Accumulators
        probSum = np.zeros((NUM_CLASSES, D, H, W), dtype=np.float32)
        countMap = np.zeros((D, H, W), dtype=np.float32)

        for d0, h0, w0 in positions:
            d1, h1, w1 = d0 + pD, h0 + pH, w0 + pW
            patch = mriNorm[d0:d1, h0:h1, w0:w1]

            # Pad boundaries
            actualD, actualH, actualW = patch.shape
            if actualD < pD or actualH < pH or actualW < pW:
                patchFull = np.zeros((pD, pH, pW), dtype=patch.dtype)
                patchFull[:actualD, :actualH, :actualW] = patch
                patch = patchFull

            # Preprocess
            mri_ch = patch[np.newaxis, np.newaxis].astype(np.float32)
            cx, cy, cz = self.createCoordinateChannels(pD, pH, pW)
            inp = np.concatenate([mri_ch, cx, cy, cz], axis=1)

            # Inference
            logits = self.session.run([self.outputName], {self.inputName: inp})[0][0]
            prob = softmax(logits, axis=0)

            # Accumulate
            probSum[:, d0:d1, h0:h1, w0:w1] = np.maximum(
                probSum[:, d0:d1, h0:h1, w0:w1],
                prob[:, :actualD, :actualH, :actualW]
            )
            countMap[d0:d0+actualD, h0:h0+actualH, w0:w0+actualW] += 1

        # Average
        countMap[countMap == 0] = 1
        seg = np.argmax(probSum / countMap, axis=0)

        return seg

    def createSegmentation(self, segArray, refVolume, labelmapVolume):
        """Create segmentation node (contains labelmap volume and segmentation node)"""
        from slicer import util
        import vtk
        
        # Get reference volume info
        refSpacing = refVolume.GetSpacing()
        refOrigin = refVolume.GetOrigin()
        
        # Set dimensions (model output is (D,H,W), directly used for Slicer)
        D, H, W = segArray.shape
        # No transpose needed, model output format is correct
        binaryMask_slicer = segArray.astype(np.uint8)
        
        # Update data (before setting matrix)
        util.updateVolumeFromArray(labelmapVolume, binaryMask_slicer)
        
        # Get IJKToRAS matrix (after updating data) - may need to re-fetch
        ijkToRasMatrix = vtk.vtkMatrix4x4()
        refVolume.GetIJKToRASMatrix(ijkToRasMatrix)
        labelmapVolume.SetIJKToRASMatrix(ijkToRasMatrix)
        labelmapVolume.SetSpacing(refSpacing)
        labelmapVolume.SetOrigin(refOrigin)
        
        # ===== Key modification: Create segmentation node and convert =====
        # Create segmentation node
        segmentation_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        segmentation_node.SetName("TN_Segmentation")
        
        # Convert labelmap to segmentation node (key step!)
        slicer.modules.segmentations.logic().ImportLabelmapToSegmentationNode(
            labelmapVolume, segmentation_node)
        
        # Set segment names and colors
        label_names = {1: "R_Trigeminal", 2: "L_Trigeminal", 3: "Vessel"}
        colors = {
            1: [1.0, 0.0, 0.0],  # Red
            2: [0.0, 1.0, 0.0],  # Green
            3: [0.0, 0.0, 1.0]   # Blue
        }
        
        for label_id, label_name in label_names.items():
            segment_id = f"Label_{label_id}"
            segment = segmentation_node.GetSegmentation().GetSegment(segment_id)
            if segment:
                segment.SetName(label_name)
                if label_id in colors:
                    segment.SetColor(colors[label_id])
        
        logging.info(f"Created segmentation node: {segmentation_node.GetName()}")
        return segmentation_node