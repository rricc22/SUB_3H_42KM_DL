# Changelog - Inferences Scripts

## v3.0 (2025-11-24) - Lag-Llama Support

### Added
- ✅ Full support for Lag-Llama Transformer model in `inference.py` and `evaluate_test.py`
- ✅ Automatic detection of `num_users` from checkpoint weights when not explicitly saved
- ✅ Automatic inference of `d_model` from checkpoint weights for Lag-Llama
- ✅ Support for all three model types: LSTM, LSTM with embeddings, and Lag-Llama

### Changed
- Updated `inference.py` to handle Lag-Llama model architecture
- Updated `evaluate_test.py` to handle Lag-Llama model architecture
- Enhanced checkpoint loading to automatically detect model parameters from weights
- Updated README.md with Lag-Llama examples and checkpoint information

### Technical Details
- Detects `num_users` from `user_embedding.weight` shape in checkpoint
- Detects `d_model` from `input_projection.weight` shape in checkpoint
- Uses default values for `nhead` (8) and `dim_feedforward` (512) if not in checkpoint args

### Testing
- ✅ Tested `inference.py` with `checkpoints/lag_llama_best.pt` - Works correctly
- ✅ Tested `evaluate_test.py` with `checkpoints/lag_llama_best.pt` - Generates visualization
- ✅ Results: MAE: 16.55 BPM, RMSE: 20.74 BPM, R²: -0.1019

### Files Modified
- `Inferences/inference.py` - Added Lag-Llama support
- `Inferences/evaluate_test.py` - Added Lag-Llama support
- `Inferences/README.md` - Updated documentation

### Backward Compatibility
- ✅ All changes are backward compatible
- ✅ LSTM and LSTM with embeddings models still work as before
- ✅ No breaking changes to API or command-line arguments

---

## v2.0 (Previous)
- Fixed dict/namespace compatibility in checkpoint loading
- Improved imports using package-style
- Better path handling with `sys.path.insert(0, ...)`
