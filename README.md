# EV Battery Inspection with Segmentation and Object Detection

End-to-end industrial computer vision pipeline for **EV battery inspection** using **object detection** and **segmentation** to locate key regions and highlight potential defects/abnormalities for automated quality control.

Project page: https://drsaqibbhatti.com/projects/ev-battery-inspection.html

---

## Overview

### What this project does
- Detects battery components / regions of interest (ROI) using **object detection**
- Extracts pixel-level masks using **segmentation** for accurate boundary/area analysis
- Produces inspection-friendly outputs:
  - bounding boxes + masks overlay
  - per-region confidence
  - optional pass/fail rules for automated decisions

### Why it matters
EV battery inspection often requires:
- consistent inspection across multiple ROIs per unit
- accurate shape/edge validation (pixel-level precision)
- robustness against reflections, glare, and surface variation

This pipeline is designed for real production-style conditions and can integrate into AOI/QC systems.

> Note: Client/product-specific details, exact defect taxonomy, and production metrics are omitted (confidential).

---

## Key Features
- Combined **detection + segmentation** workflow for industrial inspection
- Region-level and pixel-level analysis
- Robust preprocessing options for challenging images
- Post-processing filters for stable decision-making (noise removal, region constraints)
- Deployment-friendly structure (inference script produces overlays/results)

---

## Pipeline
1. **Input**: image/frame from inspection camera
2. **Preprocessing**:
   - normalization / contrast enhancement
   - denoise / glare handling (as needed)
3. **Object Detection**:
   - locate ROIs (battery edges, terminals, labels, etc.)
4. **Segmentation**:
   - generate masks for key regions (semantic/instance)
5. **Post-processing**:
   - mask cleanup (morphology)
   - rule checks (area/shape/overlap/alignment thresholds)
6. **Output**:
   - annotated overlays (boxes + masks)
   - optional structured results (JSON/CSV) for automation

---

## Tech Stack
- **Python**
- **PyTorch**
- **OpenCV**, **NumPy**

Frameworks/models may include (depending on implementation):
- Detection: YOLO / custom detector
- Segmentation: U-Net /CNN / custom segmentation network

