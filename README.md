# Advanced Machine Learning 2024/25/2 @ ELTE
### **Project**: Salsa Subgenre Detection with LSTM models
### **Authors**: Attila Barna and Balázs Menkó
---
---

### Description
We tried to build an LSTM model to separate three different salsa subgenres (Linear Salsa, Son, and Rumba). We created 3 models:
- a basic LSTM
- a Bidirectional LSTM with Self-Attention
- Enhanced LSTM with BatchNorm & Global Pooling

### Key differences:

| Feature             | Model 1            | Model 2            | Model 3            |
|---------------------|--------------------|--------------------|--------------------|
| Bidirectional LSTM  | ✗                  | ✓                  | ✓                  |
| Self-Attention      | ✗                  | ✓                  | ✓                  |
| Batch Normalization | ✗                  | ✗                  | ✓                  |
| Global Pooling      | ✗                  | ✗                  | ✓                  |
| Dense Layers        | 1                  | 1                  | 2                  |
| Batch Size          | 32                 | 16                 | 16                 |
| Return Sequences    | First layer only   | First layer only   | First layer only   |
| Max Dropout Rate    | 30%                | 30%                | 40%                |

### Extracted features:
- MFCCs Mel-frequency cepstral coefficients (Dim = 13)
- Chroma, Harmonic content and tonality (Dim = 12)
- Spectral Contrast, Frequency distribution (Dim = 7)
- Onset, Rhythmic pattern detection (Dim = 1)
- Tempogram, Tempo and rhythmic periodicity (Dim = 384)

### Result:

| Metric        | Model 1  | Model 2  | Model 3  |
|---------------|----------|----------|----------|
| Test Accuracy | 74.29%   | 57.14%   | 71.43%   |
| Test Loss     | 1.0525   | 1.2996   | 0.6219   |

