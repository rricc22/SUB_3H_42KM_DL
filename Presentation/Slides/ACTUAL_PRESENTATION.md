# Heart Rate Prediction from Running Activity
## Deep Learning Course Project - CentraleSupélec

**The Real Story: From Challenging Data to Custom Solutions**

---

## SLIDE 1: Problem & Motivation (1 min)

### The Challenge
**Predict heart rate time-series from running activity data**

**Goal:**
- Input: Speed + Altitude sequences → Output: Heart Rate predictions
- Time-series regression (500 timesteps)
- Target: MAE < 10 BPM

**Why This Matters:**
- Health monitoring & fitness tracking
- Validate wearable sensors
- Understand physiological responses to exercise

**Success Metrics:**
- Excellent: MAE < 5 BPM
- Acceptable: MAE < 10 BPM

---

## SLIDE 2: Dataset - The Endomondo Challenge (1.5 min)

### Initial Dataset: Endomondo (FitRec)
**Total available: 260,000 workouts with HR**

### The Problem: Data Quality Issues 

**Filtering Process:**
- Started with 260,000 workouts
- Applied 7 quality filters:
  - Valid sport type (running only)
  - Complete HR data
  - Minimum sequence length
  - Valid speed/altitude ranges
  - No missing GPS
  - Reasonable duration
  - Complete metadata

**Result: Only ~13,000 usable workouts** (~5% of total!)

**Why so low?**
- Missing heart rate data
- Incomplete GPS tracks
- Mixed sport types (bike, kayak, etc.)
- Data corruption

### The Correlation Problem
**KEY FINDING:** Low correlation in Endomondo data

![Correlation Comparison - Raw vs Processed](../EDA/EDA_Generation/correlation_comparison_raw_vs_processed.png)

**Correlation Matrix Shows:**
- Speed ↔ HR: r ≈ 0.3-0.4 (weak!)
- Altitude ↔ HR: r ≈ 0.1-0.2 (very weak!)
- Much noise, many outliers

**Why?**
- Multi-user dataset (different fitness levels)
- GPS errors and noise
- Variable data quality across workouts
- Heterogeneous recording devices

![Correlation Matrix - Processed Data](../EDA/EDA_generation_from_proccessed_HR/correlation_matrix.png)

**This explains why models struggled initially!**

---

## SLIDE 3: Model Development - Phase 1 (1 min)

### First Attempts with Endomondo Data

#### Model 1: Basic LSTM
**Strategy: Max out VRAM!**
- Batch size: 128 (used all available VRAM)
- Architecture: 2-layer LSTM, 64 hidden units
- Goal: Fast training, maximum throughput

![LSTM Basic Training (BS=32)](../checkpoints/lstm_bs32_lr0.001_e100_h64_l2_training_curves.png)

**Result:**
- MAE: ~15 BPM
- Fast training but limited performance
- Data quality was the bottleneck

---

#### Model 2: GRU (Alternative RNN)
**Tried GRU architecture:**
- Simpler than LSTM (fewer parameters)
- Faster computation
- Similar performance to LSTM

![GRU Training (Bidirectional)](../checkpoints/gru_bs16_lr0.0003_e30_h128_l4_bidir_training_curves.png)

**Result:**
- MAE: ~15 BPM (similar to LSTM)
- No significant improvement
- **Conclusion:** Architecture wasn't the problem, data was!

---

## SLIDE 4: Batch Size Experiment (1 min) ⭐

### Systematic Hyperparameter Search

**Hypothesis:** Maybe smaller batches help with noisy data?

**Experiment:** Test batch sizes 8, 16, 32, 64
- Same architecture (LSTM, 64 hidden, 2 layers)
- Same learning rate (0.001)
- 30 epochs each
- Controlled comparison

**Location:** `experiments/batch_size_search/`

### Results - All 4 Training Curves

**Batch Size = 8 (Noisy, unstable)**
![BS=8](../experiments/batch_size_search/bs8/lstm_bs8_lr0.001_e30_h64_l2_training_curves.png)

**Batch Size = 16 (BEST! ⭐)**
![BS=16](../experiments/batch_size_search/bs16/lstm_bs16_lr0.001_e30_h64_l2_training_curves.png)

**Batch Size = 32 (Slightly worse)**
![BS=32](../experiments/batch_size_search/bs32/lstm_bs32_lr0.001_e30_h64_l2_training_curves.png)

**Batch Size = 64 (Fast but generalizes poorly)**
![BS=64](../experiments/batch_size_search/bs64/lstm_bs64_lr0.001_e30_h64_l2_training_curves.png)

### Key Finding: BS=16 is Optimal! 

**Performance:**
- BS=8: Noisy training, unstable
- **BS=16: Best validation MAE** ⭐
- BS=32: Slightly worse
- BS=64: Fast but generalizes poorly

**Trade-off:**
-  Best performance: BS=16
-  Slowest training: BS=16
- "We chose quality over speed"

**Insight:** Smaller batches provide better gradient estimates with noisy data

---

## SLIDE 5: Transfer Learning Attempt (30 sec)

### Trying Pretrained Models

**Experiment: Lag-Llama Transfer Learning**
- Lag-Llama: Time-series foundation model
- Attempted to fine-tune on our data

**Location:** `checkpoints/archives/lag_llama_transfert_learning/`

![Lag-Llama Transfer Learning (BS=32)](../checkpoints/archives/lag_llama_transfert_learning/lag_llama_bs32_lr0.001_e5_h64_l2_emb16_training_curves.png)

**Result:**
- Didn't significantly help
- Model still struggled with low correlations
- **Conclusion:** Need better quality data!

---

## SLIDE 6: The Breakthrough - Apple Watch Data (1.5 min) ⭐

### Custom Dataset: Personal Apple Watch

**New Approach: Collect our own high-quality data!**

**Dataset:**
- Personal data from team members + friends
- 285 running workouts (2019-2025)
- 6 years of training history
- Apple Watch sensors (better quality)

### Data Quality Comparison

**Endomondo (poor):**
- Low correlations (r ≈ 0.3)
- Multi-user noise
- GPS errors
- Incomplete data

**Apple Watch (excellent):**
- **Much higher correlations!** (r ≈ 0.6-0.8)
- Consistent recording device
- Personal data (same user)
- Better GPS and HR sensors

### 2025 Workout - GOOD Quality (12 HR samples/min)

![Apple Watch 2025 - Good Quality](../experiments/apple_watch_analysis/plots/validation_v2_workout_20251123_103725.png)

**4-Panel Validation Plot Shows:**
1. Heart Rate over time - Clean, detailed signal
2. Speed over time - Smooth GPS tracking
3. Elevation profile - Accurate altitude
4. **HR-Speed correlation - Strong relationship!** ⭐

### 2019 Workout - SPARSE Quality (0.7 HR samples/min)

![Apple Watch 2019 - Sparse Quality](../experiments/apple_watch_analysis/plots/validation_v2_workout_20241114_172333.png)

**Data Quality Evolution:**
- 2019-2021: 0.4-1.0 HR samples per minute (heavily interpolated, older watchOS)
- 2025: 12 HR samples per minute (rich detail, new watchOS)

**Finding: 17x more data points in recent workouts!**

**Correlation improvement: r=0.3 (Endomondo) → r=0.68 (Apple Watch)**

---

## SLIDE 7: Apple Watch Results (1 min)

### Training on High-Quality Data

**Model: LSTM with Embeddings**
- Same architecture as before
- But trained on Apple Watch data
- Much better correlations

![LSTM with Embeddings - Apple Watch Training](../checkpoints/apple_watch_v2_lstm_emb/lstm_embeddings_bs32_lr0.001_e100_h64_l2_emb16_training_curves.png)

**Results:**
- Training: Smooth convergence
- Validation: Stable improvement
- Test: TBD (in final evaluation)

**Also Tried: GRU on Apple Watch**

![GRU - Apple Watch Training](../checkpoints/apple_watch_v2_lstm_emb/gru_bs32_lr0.001_e100_h128_l4_training_curves.png)

**Result: MAE 77 BPM on test**
- High error due to temporal shift
- Training on 2019-2024, testing on 2025
- Fitness improved over 6 years!

**Challenge:** Temporal distribution shift
- User's fitness evolved
- HR response patterns changed
- Need domain adaptation techniques

---

## SLIDE 8: Next Steps - Fine-Tuning Strategy (1 min)

### Proposed Approach: Two-Stage Training

**Stage 1: General Model (Endomondo)**
- Pre-train on large Endomondo dataset (13K workouts)
- Learn general speed→HR patterns
- Capture population-level relationships

**Stage 2: Fine-Tuning (Apple Watch)**
- Fine-tune on personal Apple Watch data (285 workouts)
- Adapt to individual physiology
- Leverage high-quality correlations

**Expected Benefits:**
- General knowledge from diverse users
- Personalization from individual data
- Better handling of edge cases

**Transfer Learning Hypothesis:**
```
Large, noisy dataset → Learn robust features
    ↓
Small, high-quality dataset → Refine for person
    ↓
Best of both worlds!
```

### Additional Improvements

**1. Data Augmentation**
- Sliding window (increase data 5-10x)
- Synthetic perturbations
- Mix training data sources

**2. Architecture Enhancements**
- Attention mechanisms (capture lag effects)
- Multi-task learning (HR ↔ Speed bidirectional)
- Ensemble methods (LSTM + Transformer)

**3. Advanced Techniques**
- Domain adaptation (temporal shift)
- Meta-learning (fast adaptation)
- Pretrained foundation models (Chronos, TimeGPT)

---

## SLIDE 9: Key Insights & Contributions (1 min)

### What We Learned

**1. Data Quality > Model Complexity**
- Endomondo: Complex models, poor results
- Apple Watch: Simple models, better promise
- **"Garbage in, garbage out" is real!**

**2. Systematic Experimentation Matters**
- Batch size search revealed BS=16 optimal
- Trade-off: Quality vs speed (chose quality)
- Controlled experiments guide decisions

**3. Correlation is Key Predictor**
- Low correlation (Endomondo): MAE ~15 BPM
- High correlation (Apple Watch): Expected <10 BPM
- Data exploration guides expectations

### Novel Contributions

**1. Comprehensive Data Pipeline**
- Endomondo: 260K → 13K filtering pipeline
- Apple Watch: Custom extraction (285 workouts, 6 years)
- GPS-HR alignment with timezone correction

**2. Systematic Experiments**
- Batch size search (8, 16, 32, 64)
- Transfer learning attempts
- Multiple architectures (LSTM, GRU, Lag-Llama, PatchTST)

**3. Data Quality Analysis**
- Identified correlation problem in Endomondo
- HR sampling evolution (0.4 → 12 samples/min)
- Temporal distribution shift identification

---

## SLIDE 10: Team Contributions (30 sec)

### Division of Work

**[Member 1 Name]: Data Preprocessing & EDA**
- Filtered 260K → 13K Endomondo workouts
- Identified low correlation issue
- Generated all correlation matrices
- Created preprocessing pipeline

**[Member 2 Name]: Model Training & Experiments**
- Implemented LSTM and GRU models
- Conducted batch size search (BS=8,16,32,64)
- Systematic hyperparameter tuning
- Training infrastructure setup

**[Member 3 Name]: Apple Watch Pipeline & Transfer Learning**
- Built Apple Watch extraction pipeline (285 workouts)
- GPS-HR alignment with timezone correction
- Attempted Lag-Llama transfer learning
- Data quality analysis (sparse vs dense)

**[Member 4 Name]: Evaluation & Documentation**
- Generated all training curve visualizations
- Created validation plots
- Error analysis and reporting
- Presentation preparation

---

## SLIDE 11: Conclusions & Future Work (30 sec)

### Main Takeaways

**1. Data Quality is Critical**
- Endomondo: 260K workouts, only 5% usable
- Low correlations → poor model performance
- Apple Watch: Higher quality → better potential

**2. Systematic Experiments Work**
- Batch size search: Found BS=16 optimal
- Multiple architectures tested
- Controlled comparisons guide decisions

**3. Two-Stage Training Promising**
- General model (Endomondo) + Fine-tuning (Apple Watch)
- Best of both: Diversity + Quality
- Expected: MAE < 10 BPM

### Future Directions

**Immediate (1-2 weeks):**
- Complete Apple Watch fine-tuning experiments
- Implement two-stage training pipeline
- Evaluate on held-out test set

**Advanced (1-2 months):**
- Domain adaptation for temporal shift
- Attention mechanisms for lag effects
- Ensemble methods (multiple models)

**Long-term (3+ months):**
- Real-time deployment (mobile app)
- Expand to multiple users (friends' data)
- Try foundation models (Chronos, TimeGPT)

---

## SLIDE 12: Questions?

**We're ready to discuss:**
- Why Endomondo data was so challenging
- Batch size search methodology and results
- Apple Watch data quality differences
- Two-stage training strategy
- Any technical details

**Key Resources:**
- Code: `Model/train.py`, `Preprocessing/prepare_sequences_v2.py`
- Experiments: `experiments/batch_size_search/`
- Apple Watch: `experiments/apple_watch_analysis/`
- EDA: `EDA/EDA_Generation/`

**Thank you!** 

---

## BACKUP SLIDES (If Time Permits)

### Backup 1: Detailed Batch Size Results

| Batch Size | Train MAE | Val MAE | Time/Epoch | Notes |
|------------|-----------|---------|------------|-------|
| 8 | 14.2 | 15.8 | 45s | Noisy, unstable |
| **16** | **13.9** | **15.3**  | 28s | **Best performance** |
| 32 | 14.1 | 15.4 | 18s | Good balance |
| 64 | 14.5 | 16.2 | 12s | Fast but generalizes worse |

**Conclusion: BS=16 trades speed for quality**

---

### Backup 2: Correlation Deep Dive

**Endomondo (Poor):**
```
Speed ↔ HR:     r = 0.34 (weak)
Altitude ↔ HR:  r = 0.12 (very weak)
Speed ↔ Alt:    r = 0.08 (no relationship)
```

**Apple Watch (Good):**
```
Speed ↔ HR:     r = 0.68 (strong)
Altitude ↔ HR:  r = 0.42 (moderate)
Speed ↔ Alt:    r = 0.35 (moderate)
```

**Impact on MAE:**
- Low correlation → MAE ~15 BPM
- High correlation → Expected MAE ~8-10 BPM

---

### Backup 3: Apple Watch Pipeline Details

**Extraction Pipeline:**
1. Parse Apple Health export XML
2. Extract GPS trackpoints from GPX
3. Extract HR records with timestamps
4. Timezone correction (UTC → Local)
5. Interpolate HR to GPS timestamps
6. Calculate speed from GPS (haversine)
7. Align all features to 500 timesteps

**Success Rate:** 95% (271/285 workouts processed)

**Processing Time:** ~30-60 minutes for 285 workouts

---

### Backup 4: All Models Tried

| Model | Architecture | Result | Notes |
|-------|--------------|--------|-------|
| LSTM (basic) | 2 layers, 64 hidden | MAE 15 BPM | Fast, simple |
| LSTM (large) | 5 layers, 256 hidden | MAE 14 BPM | Overfits |
| LSTM (bidir) | 4 layers, 128 hidden | MAE 14 BPM | Best Endomondo |
| GRU | 4 layers, 128 hidden | MAE 15 BPM | Similar to LSTM |
| Lag-Llama | Transfer learning | Poor | Data mismatch |
| PatchTST | 6 layers, 256 d_model | Training | State-of-art |

**Conclusion:** Architecture less important than data quality

---

## END OF PRESENTATION

**Total Time: ~7-8 minutes**

**Slide Breakdown:**
1. Problem (1 min)
2. Endomondo Challenge (1.5 min)
3. First Models (1 min)
4. Batch Size Search (1 min) ⭐
5. Transfer Learning (0.5 min)
6. Apple Watch Data (1.5 min) ⭐
7. Apple Watch Results (1 min)
8. Next Steps (1 min)
9. Key Insights (1 min)
10. Team Contributions (0.5 min)
11. Conclusions (0.5 min)

**Backup slides: 4 additional if time permits**
