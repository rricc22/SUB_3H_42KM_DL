# Presentation Materials - ACTUAL Version

## ✅ What's New

Based on your feedback, I've created the **ACTUAL** presentation that follows YOUR real project story:

### The Real Story:
1. **Endomondo Challenge** - 260K workouts, only 13K usable, LOW correlations
2. **Initial Models** - LSTM/GRU with large batch sizes (max VRAM)
3. **Batch Size Search** - Systematic experiment, BS=16 wins (but slower)
4. **Transfer Learning** - Tried Lag-Llama, didn't help much
5. **Apple Watch Breakthrough** - Custom high-quality data, MUCH better correlations
6. **Fine-Tuning Plan** - Two-stage training (general → personal)

---

## 📄 New Files Created

### 1. **ACTUAL_PRESENTATION.md** ⭐ USE THIS
- Follows YOUR actual project journey
- 13 slides (7 core + 4 backup)
- Story: Bad data → Experiments → Better data → Results
- Includes all your graphs in the right places

### 2. **VISUALIZATION_CHECKLIST.md** ⭐ USE THIS
- Exact paths to all 13-14 graphs
- Slide-by-slide breakdown
- Layout suggestions (2x2 grid for batch size, etc.)
- Speaking points for key graphs

### 3. Other files (generic versions)
- `PROJECT_PRESENTATION.md` - Generic version
- `SPEAKER_NOTES.md` - Generic notes
- `SLIDE_DECK.md` - Generic slides
- `PRESENTATION_SUMMARY.md` - Generic summary

**Use ACTUAL_PRESENTATION.md + VISUALIZATION_CHECKLIST.md**

---

## 🎯 Your Story in 7 Minutes

### Timing Breakdown

| Slide | Topic | Time | Key Point |
|-------|-------|------|-----------|
| 1 | Problem | 1:00 | Predict HR from activity |
| 2 | Endomondo Challenge | 1:30 | 260K→13K, low correlations ⚠️ |
| 3 | First Models | 1:00 | LSTM/GRU, MAE ~15 BPM |
| 4 | Batch Size Search | 1:00 | BS=16 best (show all 4 graphs!) ⭐ |
| 5 | Transfer Learning | 0:30 | Lag-Llama didn't help |
| 6 | Apple Watch | 1:30 | High quality, r=0.68 ⭐ |
| 7 | Results | 1:00 | Training curves |
| 8 | Next Steps | 1:00 | Fine-tuning strategy |
| 9-12 | Wrap-up | 1:00 | Insights, team, conclusions |

**Total: 8:30 minutes** (gives 1:30 buffer for Q&A)

---

## 📊 Key Graphs to Include

### Must-Have (8 graphs):

1. **Endomondo correlations** (2 graphs)
   - Shows low correlation problem (r ≈ 0.3)
   - Slide 2

2. **Batch size search** (4 graphs) ⭐
   - ALL 4 curves: BS=8, 16, 32, 64
   - 2x2 grid layout
   - Highlight BS=16 as winner
   - Slide 4

3. **Apple Watch comparison** (2 graphs) ⭐
   - Good quality (2025): 12 samples/min
   - Sparse quality (2019): 0.7 samples/min
   - Side-by-side
   - Slide 6

### Nice-to-Have (5 graphs):

4. LSTM/GRU training curves (Slide 3)
5. Lag-Llama transfer learning (Slide 5)
6. Apple Watch training results (Slide 7)

**Total: 13 graphs**

All paths listed in `VISUALIZATION_CHECKLIST.md`

---

## 🚀 Quick Start

### Step 1: Read Your Story (15 min)
```bash
# Read the actual presentation
cat Presentation/ACTUAL_PRESENTATION.md | less

# Check graph locations
cat Presentation/VISUALIZATION_CHECKLIST.md | less
```

### Step 2: Gather Graphs (15 min)
```bash
# Copy all graphs to a temp folder for easy access
cd /home/riccardo/Documents/Collaborative-Projects/SUB_3H_42KM_DL

# Create temp folder
mkdir -p Presentation/graphs_for_slides

# Copy key graphs
cp experiments/batch_size_search/bs*/lstm_*_training_curves.png Presentation/graphs_for_slides/
cp experiments/apple_watch_analysis/plots/validation_v2_workout_202511*.png Presentation/graphs_for_slides/
cp experiments/apple_watch_analysis/plots/validation_v2_workout_20241114*.png Presentation/graphs_for_slides/
cp EDA/EDA_Generation/correlation_*.png Presentation/graphs_for_slides/
cp EDA/EDA_generation_from_proccessed_HR/correlation_matrix.png Presentation/graphs_for_slides/
cp checkpoints/lstm_bs32_lr0.001_e100_h64_l2_training_curves.png Presentation/graphs_for_slides/
cp checkpoints/gru_bs16_lr0.0003_e30_h128_l4_bidir_training_curves.png Presentation/graphs_for_slides/
```

### Step 3: Create Slides (30 min)
1. Open PowerPoint or Google Slides
2. Copy content from `ACTUAL_PRESENTATION.md`
3. Insert graphs using paths from `VISUALIZATION_CHECKLIST.md`
4. **Key layouts:**
   - Slide 4: 2x2 grid for batch size search
   - Slide 6: Side-by-side for Apple Watch comparison

### Step 4: Practice (1 hour)
1. Run through with timer
2. Practice transitions
3. Focus on key messages:
   - "Only 5% of Endomondo data usable"
   - "BS=16 optimal but slower"
   - "17x more data in 2025 workouts"

---

## 💡 Key Messages

### What Makes Your Story Unique

**1. Data Quality Matters More Than Model**
- Endomondo: 260K workouts → 13K usable → Still poor results
- Apple Watch: 285 workouts → Better results potential
- **Message: "Garbage in, garbage out"**

**2. Systematic Experiments**
- Batch size search: 8, 16, 32, 64
- Found BS=16 optimal (traded speed for quality)
- **Message: "We prioritized quality over speed"**

**3. Evolution & Adaptation**
- Started with public dataset (struggled)
- Built custom pipeline (breakthrough)
- Plan two-stage training (smart strategy)
- **Message: "When plan A fails, adapt!"**

---

## 🎤 Speaking Points

### Slide 2 (Endomondo Challenge):
> "We started with the Endomondo dataset - 260,000 workouts! Sounds great, right? But after applying quality filters, only 13,000 were usable - that's just 5%. Even worse, the correlations were terrible. Look at this matrix - speed to heart rate is only 0.34. This low correlation explained why our models struggled."

### Slide 4 (Batch Size Search):
> "We hypothesized that smaller batches might help with noisy data, so we systematically tested 4 batch sizes. Here are all the training curves. Batch size 16, shown here in the top right, achieved the best validation performance. It's slower to train, but we chose quality over speed. This is a key finding - with noisy data, smaller batches help."

### Slide 6 (Apple Watch Breakthrough):
> "This is our breakthrough. We built a custom pipeline to extract data from Apple Watches - our own and friends'. The difference is dramatic. On the left, a 2025 workout with 12 heart rate samples per minute. On the right, a 2019 workout with 0.7 samples per minute. That's 17 times more data! More importantly, correlations jumped from 0.3 to 0.68. Look at that HR-speed correlation plot - now we have a relationship to model!"

---

## ⚠️ Important Notes

### What Changed from Generic Version

**Old story (generic):**
- Started with good data
- Models worked reasonably well
- Focus on architecture

**YOUR story (actual):**
- Started with BAD data (260K → 13K, low correlations)
- Models struggled (MAE ~15 BPM)
- Systematic experiments (batch size search)
- Built custom pipeline (Apple Watch)
- Found better data (higher correlations)
- **Focus on data quality > model complexity**

### Why Your Story is Better

1. **More honest** - Shows real challenges
2. **More interesting** - Problem solving journey
3. **More impressive** - Custom data pipeline
4. **More insights** - Data quality matters
5. **Better pedagogy** - Teaches adaptation

---

## 📋 Pre-Presentation Checklist

### Content
- [ ] Read `ACTUAL_PRESENTATION.md` fully
- [ ] Understand the story arc
- [ ] Know your key messages (data quality, BS=16, Apple Watch)

### Visuals
- [ ] All 13 graphs collected
- [ ] Batch size: 2x2 grid on Slide 4
- [ ] Apple Watch: Side-by-side on Slide 6
- [ ] Green highlight on BS=16 graph

### Practice
- [ ] Practice with timer (aim 7:30, max 8:00)
- [ ] Practice transitions
- [ ] Prepare Q&A answers

### Equipment
- [ ] Laptop charged
- [ ] HDMI adapter
- [ ] Backup USB with slides + graphs
- [ ] Arrive 15 min early

---

## 🎯 Success Criteria

### Minimum
- [ ] Tell the complete story (data challenge → solution)
- [ ] Show batch size search (all 4 graphs)
- [ ] Show Apple Watch comparison
- [ ] Stay under 8 minutes

### Target
- [ ] Emphasize data quality > model complexity
- [ ] Explain systematic experimentation value
- [ ] Connect to two-stage training plan
- [ ] Engage audience with visuals

### Exceptional
- [ ] Memorable insights ("5% usable", "17x more data")
- [ ] Smooth delivery with confidence
- [ ] Strong Q&A responses
- [ ] Leave lasting impression

---

## 📞 Questions You Should Be Ready For

**Q1: "Why was Endomondo data so bad?"**
**A:** "Multi-user dataset with heterogeneous devices, GPS errors, and mixed sport types. After filtering for running with complete HR data, only 5% was usable. Even then, correlations were weak (r=0.3) due to individual variability."

**Q2: "How did you decide on BS=16?"**
**A:** "Systematic controlled experiment. We tested 8, 16, 32, 64 with identical models. BS=16 showed best validation MAE. The trade-off was training speed - it's slower but we prioritized quality."

**Q3: "What's the correlation in Apple Watch data?"**
**A:** "Speed-HR correlation jumped from 0.3 in Endomondo to 0.68 in Apple Watch. This is because it's personal data from consistent devices, not mixed users. Plus, 2025 workouts have 12 HR samples per minute versus 0.7 in 2019."

**Q4: "What's your expected final performance?"**
**A:** "With two-stage training - pre-train on Endomondo (general patterns) then fine-tune on Apple Watch (personal data) - we expect MAE under 10 BPM, possibly approaching 5 BPM on high-quality 2025 data."

---

## ✅ Final Checklist

**Files to use:**
- ✅ `ACTUAL_PRESENTATION.md` (main content)
- ✅ `VISUALIZATION_CHECKLIST.md` (graph paths)

**Key slides:**
- ✅ Slide 2: Endomondo challenge (low correlations)
- ✅ Slide 4: Batch size search (2x2 grid) ⭐
- ✅ Slide 6: Apple Watch breakthrough (side-by-side) ⭐

**Key messages:**
- ✅ Data quality > model complexity
- ✅ Systematic experiments guide decisions
- ✅ Adaptation when faced with challenges

---

**You've got a great story to tell! Good luck! 🚀**

**Status:** ✅ Ready for presentation
**Next:** Read ACTUAL_PRESENTATION.md → Gather graphs → Create slides → Practice
