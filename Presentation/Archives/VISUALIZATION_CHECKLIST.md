# Visualization Checklist - Actual Presentation

## 📊 Graphs to Include (In Order)

### SLIDE 2: Endomondo Data Challenge

**Graph 1: Correlation Comparison (Raw vs Processed)**
```
Path: EDA/EDA_Generation/correlation_comparison_raw_vs_processed.png
Message: "Low correlations in Endomondo data (r ≈ 0.3-0.4)"
Position: Full slide or half-slide
```

**Graph 2: Correlation Matrix (Processed Data)**
```
Path: EDA/EDA_generation_from_proccessed_HR/correlation_matrix.png
Message: "Even after preprocessing, correlations remain weak"
Position: Side-by-side with Graph 1
```

---

### SLIDE 3: First Model Attempts

**Graph 3: LSTM Basic Training (BS=32)**
```
Path: checkpoints/lstm_bs32_lr0.001_e100_h64_l2_training_curves.png
Message: "Initial LSTM with large batch size, MAE ~15 BPM"
Position: Half slide
```

**Graph 4: GRU Training (Bidirectional)**
```
Path: checkpoints/gru_bs16_lr0.0003_e30_h128_l4_bidir_training_curves.png
Message: "GRU alternative, similar performance"
Position: Half slide, next to LSTM
```

---

### SLIDE 4: Batch Size Search ⭐ KEY SLIDE

**Graph 5-8: All 4 Batch Size Comparisons**
```
Layout: 2x2 grid

Top-left:    experiments/batch_size_search/bs8/lstm_bs8_lr0.001_e30_h64_l2_training_curves.png
Top-right:   experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_training_curves.png
Bottom-left: experiments/batch_size_search/bs32/lstm_bs32_lr0.001_e30_h64_l2_training_curves.png
Bottom-right: experiments/batch_size_search/bs64/lstm_bs64_lr0.001_e30_h64_l2_training_curves.png

Message: "BS=16 shows best validation performance (highlight this one!)"
Position: Full slide, 2x2 grid
```

**Key Points to Highlight:**
- BS=8: Noisy, unstable
- **BS=16: Best val MAE** ← PUT GREEN BORDER OR ARROW
- BS=32: Slightly worse
- BS=64: Fast but generalizes poorly

---

### SLIDE 5: Transfer Learning Attempt

**Graph 9: Lag-Llama Transfer (BS=32)**
```
Path: checkpoints/archives/lag_llama_transfert_learning/lag_llama_bs32_lr0.001_e5_h64_l2_emb16_training_curves.png
Message: "Transfer learning didn't significantly help with noisy data"
Position: Full slide or half
```

**Alternative/Additional:**
```
Path: checkpoints/archives/lag_llama_transfert_learning/lag_llama_bs64_lr0.001_e5_h64_l2_emb16_training_curves.png
Message: "Different batch sizes, similar results"
```

---

### SLIDE 6: Apple Watch - The Breakthrough ⭐ KEY SLIDE

**Graph 10: Good Quality Workout (2025)**
```
Path: experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png
Message: "Recent 2025 data: 12 HR samples/min, strong correlations"
Position: Half slide (left)
4-Panel shows: HR, Speed, Elevation, HR-Speed correlation
```

**Graph 11: Sparse Quality Workout (2019)**
```
Path: experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png
Message: "Older 2019 data: 0.7 HR samples/min, interpolated"
Position: Half slide (right), side-by-side comparison
```

**Key Message:** 
"17x more data points in 2025 workouts!"
"Correlation improved from r=0.3 (Endomondo) to r=0.68 (Apple Watch)"

---

### SLIDE 7: Apple Watch Results

**Graph 12: LSTM with Embeddings (Apple Watch)**
```
Path: checkpoints/apple_watch_v2_lstm_emb/lstm_embeddings_bs32_lr0.001_e100_h64_l2_emb16_training_curves.png
Message: "Training on high-quality Apple Watch data"
Position: Half slide
```

**Graph 13: GRU (Apple Watch)**
```
Path: checkpoints/apple_watch_v2_lstm_emb/gru_bs32_lr0.001_e100_h128_l4_training_curves.png
Message: "GRU results, shows temporal shift challenge (MAE 77 BPM)"
Position: Half slide, next to LSTM
```

---

### SLIDE 10: Visualization Summary (Optional Grid)

**If time permits, show overview grid:**
- Top row: Data quality comparison (Endomondo vs Apple Watch)
- Middle row: Batch size search results (all 4)
- Bottom row: Best models (LSTM bidir, Apple Watch training)

---

## 📋 Complete Graph Checklist

Use this to verify you have all graphs ready:

- [ ] **Slide 2:** Endomondo correlation plots (2 graphs)
- [ ] **Slide 3:** LSTM + GRU training curves (2 graphs)
- [ ] **Slide 4:** Batch size search - ALL 4 graphs (BS=8,16,32,64) ⭐
- [ ] **Slide 5:** Lag-Llama transfer learning (1-2 graphs)
- [ ] **Slide 6:** Apple Watch validation - Good vs Sparse (2 graphs) ⭐
- [ ] **Slide 7:** Apple Watch training results (2 graphs)

**Total: 13-14 graphs**

---

## 🎨 Formatting Tips

### For PowerPoint/Google Slides:

1. **Slide 4 (Batch Size) Layout:**
   ```
   +----------------------------------+
   | Batch Size Search Results        |
   +----------------------------------+
   |  BS=8        |  BS=16 ✅        |
   |  [curve]     |  [curve]         |
   +---------------+------------------+
   |  BS=32       |  BS=64           |
   |  [curve]     |  [curve]         |
   +----------------------------------+
   ```
   - Add green border around BS=16
   - Add text: "Best Performance ✅"

2. **Slide 6 (Apple Watch) Layout:**
   ```
   +----------------------------------+
   | Apple Watch Data Quality         |
   +----------------------------------+
   | 2025 (Good)    | 2019 (Sparse)   |
   | 12 samples/min | 0.7 samples/min |
   |                |                 |
   | [4-panel plot] | [4-panel plot]  |
   +----------------+-----------------+
   ```

3. **Image Settings:**
   - Use PNG format (best quality)
   - Insert at ~50-75% slide width for single images
   - For grids: 40-45% width each
   - Maintain aspect ratio
   - Add captions below each image

---

## 🎤 Speaking Points for Key Graphs

### Batch Size Search (Slide 4):
> "We systematically tested 4 different batch sizes. Looking at these training curves, you can see batch size 16 achieves the best validation performance, shown here in the top right. While it's slower to train, we chose quality over speed. Batch sizes 8 and 64 show worse generalization."

### Apple Watch Comparison (Slide 6):
> "This comparison shows our breakthrough. On the left, a recent 2025 workout with 12 heart rate samples per minute - notice the detailed, realistic fluctuations in the HR curve and the strong correlation with speed. On the right, a 2019 workout with only 0.7 samples per minute - heavily interpolated, smooth lines. The difference is dramatic: 17 times more data points and correlations improved from 0.3 to 0.68."

---

## ⚡ Quick Access Paths

Copy these for easy access during presentation prep:

```bash
# Endomondo correlations
EDA/EDA_Generation/correlation_comparison_raw_vs_processed.png
EDA/EDA_generation_from_proccessed_HR/correlation_matrix.png

# Basic models
checkpoints/lstm_bs32_lr0.001_e100_h64_l2_training_curves.png
checkpoints/gru_bs16_lr0.0003_e30_h128_l4_bidir_training_curves.png

# Batch size search (ALL 4)
experiments/batch_size_search/bs8/lstm_bs8_lr0.001_e30_h64_l2_training_curves.png
experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_training_curves.png
experiments/batch_size_search/bs32/lstm_bs32_lr0.001_e30_h64_l2_training_curves.png
experiments/batch_size_search/bs64/lstm_bs64_lr0.001_e30_h64_l2_training_curves.png

# Transfer learning
checkpoints/archives/lag_llama_transfert_learning/lag_llama_bs32_lr0.001_e5_h64_l2_emb16_training_curves.png

# Apple Watch validation
experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png
experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png

# Apple Watch training
checkpoints/apple_watch_v2_lstm_emb/lstm_embeddings_bs32_lr0.001_e100_h64_l2_emb16_training_curves.png
checkpoints/apple_watch_v2_lstm_emb/gru_bs32_lr0.001_e100_h128_l4_training_curves.png
```

---

## ✅ Pre-Presentation Checklist

- [ ] All 13-14 graphs copied to presentation folder
- [ ] Graphs inserted into correct slides
- [ ] Batch size search in 2x2 grid (Slide 4)
- [ ] Apple Watch comparison side-by-side (Slide 6)
- [ ] Green highlight on BS=16 graph
- [ ] Captions added to all graphs
- [ ] Images readable from back of room (test!)
- [ ] Backup USB with all graphs

---

**Key Slides with Graphs:**
- **Slide 2:** Endomondo problem (2 graphs)
- **Slide 4:** Batch size search ⭐ (4 graphs)
- **Slide 6:** Apple Watch breakthrough ⭐ (2 graphs)

**These 8 graphs tell your story!**

Good luck! 🚀
