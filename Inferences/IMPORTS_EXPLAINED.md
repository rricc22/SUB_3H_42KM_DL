# Import Styles Explained

##  What Changed

###  Old Approach (Hacky)
```python
# Add Model directory to path
sys.path.append(str(Path(__file__).parent.parent / "Model"))

from LSTM import HeartRateLSTM
from LSTM_with_embeddings import HeartRateLSTMWithEmbeddings
```

###  New Approach (Clean & Professional)
```python
# Add project root to path for clean imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from Model.LSTM import HeartRateLSTM
from Model.LSTM_with_embeddings import HeartRateLSTMWithEmbeddings
```

---

##  Why the Change?

### **1. Explicit Package Structure**
- **Old:** `from LSTM import ...` - Where is LSTM from? Unclear!
- **New:** `from Model.LSTM import ...` - Clear! It's from the Model package

### **2. Better Organization**
- Makes it obvious that models live in `Model/` directory
- Easier to understand for new developers
- Follows Python package conventions

### **3. Avoids Name Collisions**
- **Old:** If you have another file named `LSTM.py` elsewhere, conflict!
- **New:** Explicitly says "use LSTM from Model package"

### **4. IDE Support**
- Better autocomplete in VSCode/PyCharm
- Jump-to-definition works better
- Static analysis tools understand it

### **5. Scalability**
When project grows:
```python
# Easy to organize
from Model.LSTM import HeartRateLSTM
from Model.transformers import HeartRateTransformer
from Preprocessing.transforms import Normalize
from utils.metrics import compute_mae
```

Instead of:
```python
# Confusing - where is each from?
from LSTM import HeartRateLSTM
from transformers import HeartRateTransformer
from transforms import Normalize
from metrics import compute_mae
```

---

##  Technical Details

### **sys.path.append vs sys.path.insert(0, ...)**

**append (old):**
```python
sys.path.append(str(Path(...)))  # Adds to END of search path
```
- Searched LAST
- Other packages take priority
- Can cause weird conflicts

**insert(0, ...) (new):**
```python
sys.path.insert(0, str(project_root))  # Adds to START of search path
```
- Searched FIRST
- Your code takes priority
- More predictable behavior

### **Why Add Project Root Instead of Model/?**

Adding project root allows:
```python
from Model.LSTM import HeartRateLSTM          #  Works
from Preprocessing.dataset import Dataset      #  Works
from utils.config import load_config           #  Works
```

Adding just Model/ only allows:
```python
from LSTM import HeartRateLSTM                 #  Works
from Preprocessing.dataset import Dataset      #  Fails!
from utils.config import load_config           #  Fails!
```

---

##  Best Practices Summary

### **Current Project (Scripts-based)**
 What we're doing now:
```python
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from Model.LSTM import HeartRateLSTM
```

### **Ideal (Installable Package)**
⭐ What you should do eventually:
```python
# After creating setup.py and running: pip install -e .
from heart_rate_prediction.models.lstm import HeartRateLSTM
from heart_rate_prediction.data.dataset import WorkoutDataset
```

**No sys.path hacks needed!** Python knows where to find everything.

---

##  How to Check Import Style

### **Test 1: Does it work?**
```bash
cd Project/
python3 -c "from Model.LSTM import HeartRateLSTM; print(' Works!')"
```

### **Test 2: Is the path correct?**
```bash
python3 -c "import sys; from pathlib import Path; print(Path(__file__).parent.parent)"
```

### **Test 3: Full inference test**
```bash
python3 Inferences/inference.py \
  --checkpoint checkpoints/lstm_best.pt \
  --data DATA/processed/test.pt \
  --device cpu
```

---

## 📝 When to Use Each Style

| Scenario | Import Style | Example |
|----------|-------------|---------|
| **Script in same dir** | Relative import | `from .lstm import LSTM` |
| **Script in subfolder** | Package-style | `from Model.LSTM import ...` |
| **Installed package** | Full path | `from heart_rate.models import ...` |
| **Quick test/debug** | sys.path hack | `sys.path.append(...)`  |

---

##  Next Steps (Optional Improvements)

### **1. Create `Model/__init__.py`**
```python
# Model/__init__.py
from .LSTM import HeartRateLSTM
from .LSTM_with_embeddings import HeartRateLSTMWithEmbeddings

__all__ = ['HeartRateLSTM', 'HeartRateLSTMWithEmbeddings']
```

Then import becomes:
```python
from Model import HeartRateLSTM  # Even cleaner!
```

### **2. Create `setup.py` (Make it a real package)**
```python
# setup.py
from setuptools import setup, find_packages

setup(
    name='heart-rate-prediction',
    packages=find_packages(),
)
```

Install in editable mode:
```bash
pip install -e .
```

Then:
```python
# No sys.path needed!
from heart_rate_prediction.models import HeartRateLSTM
```

---

##  Summary

**What we fixed:**
1.  Changed from `sys.path.append` to `sys.path.insert(0, ...)`
2.  Changed from `from LSTM import` to `from Model.LSTM import`
3.  Added project root instead of Model directory
4.  Made imports more explicit and professional

**Benefits:**
-  Clearer code structure
- 🛡️ Avoids name collisions
-  Better IDE support
-  Scales better as project grows
- 👥 Easier for collaborators

**Current status:**  All inference scripts working with improved imports!
