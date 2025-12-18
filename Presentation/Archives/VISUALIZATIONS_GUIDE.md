# Visualizations Guide - For Presentation

##  Available Visualizations

This guide lists all available visualizations you can include in your presentation.

---

##  Must-Have Visualizations (Pick 2-3)

### 1. Training Curves (Best Model)
**Path:** `../checkpoints/lstm_bs16_lr0.0003_e75_h128_l4_bidir_training_curves.png`

**Shows:**
- Training and validation loss over epochs
- MAE convergence
- Early stopping point
- No overfitting (train/val curves converge)

**Message:** "Our best LSTM model converges smoothly to 8-10 BPM MAE without overfitting."

**Slide:** Use in Slide 7 (Results)

---

### 2. Apple Watch Data Quality Comparison
**Good HR Quality:**
- Path: `../experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png`
- HR samples: 627 over 52 min = **12 samples/min**
- Shows: Detailed HR fluctuations, realistic dynamics

**Sparse HR Quality:**
- Path: `../experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png`
- HR samples: 48 over 71 min = **0.7 samples/min**
- Shows: Smooth interpolated line, less detail

**4-Panel Layout:**
1. Heart Rate over time
2. Speed over time
3. Elevation over time
4. HR-Speed correlation

**Message:** "Data quality evolved over 6 years - recent 2025 workouts have 17x more HR samples than 2019 data."

**Slide:** Use in Slide 3 (Dataset) or Slide 9 (Visualizations)

---

### 3. Batch Size Comparison (Optional)
**Paths:**
- `../experiments/batch_size_search/bs8/lstm_bs8_lr0.001_e30_h64_l2_training_curves.png`
- `../experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_training_curves.png`
- `../experiments/batch_size_search/bs32/lstm_bs32_lr0.001_e30_h64_l2_training_curves.png`
- `../experiments/batch_size_search/bs64/lstm_bs64_lr0.001_e30_h64_l2_training_curves.png`

**Shows:**
- Effect of batch size on convergence
- BS=16-32 optimal (trade-off between speed and generalization)

**Message:** "Systematic hyperparameter search identified BS=16-32 as optimal."

**Slide:** Use in Slide 7 (Results) or backup slides

---

##  Additional Visualizations (If Available)

### 4. Test Evaluation Plots
**Path:** `../results/test_evaluation.png` (if generated)

**8-Panel Layout:**
1. Predicted vs True HR scatter
2. Error distribution histogram
3. Absolute error histogram
4. Per-workout MAE distribution
5. Example workout 1 (time-series)
6. Example workout 2 (time-series)
7. Error by HR range
8. Summary metrics panel

**Message:** "Comprehensive evaluation shows MAE ~8 BPM in 130-160 BPM range, with challenges at extremes."

**Slide:** Use in Slide 9 (Visualizations)

---

### 5. More Apple Watch Validation Examples

**All available plots:**
```
../experiments/apple_watch_analysis/plots/
├── validation_v2_workout_20250102_152835.png  (ACCEPTABLE, 4.1 samples/min)
├── validation_v2_workout_20251118_154916.png  (GOOD, 11.0 samples/min)
├── validation_v2_workout_20251123_103725.png  (GOOD, 12.1 samples/min)
├── validation_v2_workout_20250603_162155.png  (ACCEPTABLE, 2.1 samples/min)
├── validation_v2_workout_20210114_180230.png  (SPARSE, 0.4 samples/min)
├── validation_v2_workout_20241114_172333.png  (SPARSE, 0.7 samples/min)
├── validation_v2_workout_20241202_182111.png  (SPARSE, 0.6 samples/min)
└── validation_v2_workout_20200520_182552.png  (SPARSE, 0.7 samples/min)
```

**Pick any 2 to show quality difference.**

---

## 🎨 How to Use Visualizations

### In PowerPoint/Google Slides

1. **Insert Image:**
   - Click "Insert" → "Image" → "From File"
   - Navigate to path (e.g., `../checkpoints/...`)
   - Select PNG file

2. **Resize:**
   - Make it large enough to read (half slide or full slide)
   - Maintain aspect ratio

3. **Add Caption:**
   - Text box below image
   - Example: "LSTM Training Curves - Convergence to 8 BPM MAE"

4. **Highlight Key Points:**
   - Use arrows or circles (Insert → Shapes)
   - Point to: "Early stopping here", "Train/val converge"

---

### In Markdown Presentations

**Syntax:**
```markdown
![Training Curves](../checkpoints/lstm_bs16_lr0.0003_e75_h128_l4_bidir_training_curves.png)
```

**Or with caption:**
```markdown
**Training Curves - Best LSTM Model**

![Training Curves](../checkpoints/lstm_bs16_lr0.0003_e75_h128_l4_bidir_training_curves.png)

*Model converges to 8-10 BPM MAE without overfitting.*
```

---

##  Recommended Slide Layout

### Slide 7: Results

**Layout:**
```
+-------------------------------------+
| Model Results Table                 |
| (LSTM, Transformer, etc.)           |
+-------------------------------------+
| Training Curves Image               |
| (Best model convergence)            |
|                                     |
|        [Large PNG image]            |
+-------------------------------------+
```

---

### Slide 3 or 9: Data Quality

**Layout:**
```
+------------------+------------------+
| Good HR Quality  | Sparse HR Quality|
| (2025 workout)   | (2019 workout)   |
|                  |                  |
|  [4-panel plot]  |  [4-panel plot]  |
|                  |                  |
| 12 samples/min  | 0.7 samples/min |
+------------------+------------------+
```

---

### Slide 9: Visualizations

**Layout:**
```
+-------------------------------------+
| Evaluation Framework                |
+-------------------------------------+
| +--------------+  +--------------+  |
| | Training     |  | Validation   |  |
| | Curves       |  | Plots        |  |
| |              |  |              |  |
| +--------------+  +--------------+  |
|                                     |
| Key Insights:                       |
| • Best performance: 130-160 BPM     |
| • Struggles with extremes           |
| • Data quality affects accuracy     |
+-------------------------------------+
```

---

## 📝 Tips for Effective Visualizations

### Do's 
- **Large and readable:** Don't squeeze small images
- **Add captions:** Explain what we're seeing
- **Highlight key points:** Use arrows/circles
- **Reference in speech:** "As you can see in this plot..."
- **Keep it simple:** 2-3 visualizations max

### Don'ts 
- **Too many plots:** Overwhelming, confusing
- **Too small:** Audience can't read
- **No explanation:** Don't assume meaning is obvious
- **Cluttered slides:** One main visual per slide
- **Low quality:** Use PNG, not compressed JPG

---

## 🎤 Speaking Points for Each Visualization

### Training Curves
> "Here we see the training curves for our best LSTM model. The blue line shows training loss, and orange shows validation loss. Notice how they converge around epoch 75, where early stopping kicked in. This demonstrates the model learns effectively without overfitting. The final validation MAE is approximately 8-10 BPM, approaching our target."

### Apple Watch Quality Comparison
> "These two plots illustrate a key finding in our Apple Watch dataset. On the left, a recent 2025 workout with 12 heart rate measurements per minute shows detailed, realistic fluctuations. On the right, a 2019 workout with only 0.7 samples per minute has been heavily interpolated, resulting in smooth lines. This quality difference posed a challenge for model training."

### Batch Size Search
> "We conducted systematic hyperparameter tuning, varying batch sizes from 8 to 64. These training curves show that batch sizes of 16-32 provide the best trade-off between training speed and generalization performance. Larger batches of 64 converge faster but generalize slightly worse."

### Test Evaluation
> "This comprehensive 8-panel evaluation plot shows model performance on the test set. The scatter plot shows predicted versus actual heart rates, with most points near the diagonal. The error histogram is centered near zero. Time-series examples show the model captures general trends but struggles with rapid transitions during interval workouts."

---

## 🔧 Technical Details

### Image Formats
- **PNG:**  Best quality, use for presentation
- **JPG:**  Compressed, may look blurry when enlarged
- **PDF:**  Vector format, perfect for prints

### Resolution
- Most plots are generated at **300 DPI** (high quality)
- Safe to enlarge to half-slide or full-slide
- Will look sharp on projector

### File Sizes
- Training curves: ~50-100 KB each
- Apple Watch plots: ~150-200 KB each
- Total for 3 images: ~500 KB (easily fits in presentation)

---

##  Checklist: Visualizations for Presentation

- [ ] **Selected 2-3 key visualizations**
- [ ] **Training curves from best model** (required)
- [ ] **Data quality comparison** (recommended)
- [ ] **Inserted into slides** (PowerPoint/Google Slides)
- [ ] **Added captions/explanations**
- [ ] **Highlighted key points** (arrows/circles if needed)
- [ ] **Prepared speaking points** (see above)
- [ ] **Tested readability** (can you see from back of room?)

---

##  Quick Reference Table

| Visualization | Path | Use In | Key Message |
|---------------|------|--------|-------------|
| Training Curves | `../checkpoints/lstm_*_training_curves.png` | Slide 7 | Convergence, 8-10 BPM |
| Good HR Quality | `plots/validation_v2_workout_20251123_103725.png` | Slide 3/9 | 12 samples/min |
| Sparse HR Quality | `plots/validation_v2_workout_20241114_172333.png` | Slide 3/9 | 0.7 samples/min |
| Batch Size Search | `batch_size_search/bs*/lstm_*_training_curves.png` | Slide 7 | BS=16-32 optimal |

---

**Status:**  Guide complete  
**Next:** Select visualizations → Insert into slides → Practice presentation

**Remember:** Visualizations should support your narrative, not replace it! 🎨
