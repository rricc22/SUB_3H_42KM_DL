Endomondo Dataset Structure Explanation
You have two complementary JSON files from the Endomondo fitness tracking platform:
1. endomondoHR.json (253,020 workouts)
Contains detailed time-series data WITH heart rate information. Each line is one workout with:
Time-series data (arrays synchronized by index):
- speed: Speed measurements (km/h)
- altitude: Elevation data (meters)
- heart_rate: Heart rate (BPM) - KEY for fraud detection!
- latitude / longitude: GPS coordinates
- timestamp: Unix timestamps for each measurement
Metadata:
- id: Unique workout ID
- userId: User identifier
- gender: User gender
- sport: Activity type (run, bike, bike transport, etc.)
- url: Link to original workout
Perfect for your project! Has running, cycling, and even "bike (transport)" labeled.
---
2. endomondoMeta.json (962,190 workouts)
Contains summary statistics for workouts (may not have heart rate):
Aggregate metrics:
- sport: Activity type (mountain bike, bike, run, etc.)
- duration: Total time (seconds)
- distance: Total distance (km)
- ascent / descent: Elevation gain/loss
- calories: Estimated calories burned
- timestamp: Array of timestamps (if GPS available)
Metadata:
- userId, gender, url, weather
---
Key Insights for Your Project
Advantages:
✅ Pre-labeled transport modes: Contains "bike", "bike (transport)", "run" - you can use these as fraud labels!
✅ Heart rate data: The endomondoHR file has BPM which is CRUCIAL for detecting fraud:
   - Real runners: HR correlates with speed/elevation
   - Fake (bike/scooter): HR too low or inconsistent with speed
✅ Rich GPS data: Speed, elevation, GPS coordinates for trajectory analysis
✅ Large dataset: 250K+ workouts with HR, nearly 1M total
Your Fraud Detection Approach:
1. Binary classification:
   - Genuine runs: Activities labeled as "run"
   - Fraud: Activities labeled as "bike" or "bike (transport)"
2. Key fraud indicators available in this data:
   - Speed patterns (bikes = more consistent)
   - Heart rate vs. speed correlation
   - Elevation response (runners slow down more on hills)
   - GPS trajectory smoothness
3. Simple start: Use endomondoHR.json since it has heart rate AND labeled transport modes
Next Steps:
Would you like me to help you:
1. Create a data exploration notebook to visualize the differences between running/cycling?
2. Build a simple baseline classifier using statistical features?
3. Design the deep learning architecture (LSTM/CNN for sequences)?
4. Write the one-page project description for your deliverable?
This dataset is excellent for your project! The labeled transport modes essentially give you ground truth for fraud detection.
