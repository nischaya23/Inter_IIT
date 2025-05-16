# Zelta-High-Prep-13.0

This repository contains the final submission for the Zelta High Prep 13.0 competition. The folder `TEAM_30_ZELTA_HPPS` includes the following files:

1. **`main_1_btc.py`** - Strategy for the BTC market.
2. **`main_1_eth.py`** - Strategy for the ETH market.
3. **`requirements.txt`** - Dependencies required to run the scripts.
4. **`README.md`** - Documentation and instructions.

---

## Setup Instructions

### Step 1: Clone and Install Dependencies

1. Navigate to the root directory:
   ```bash
   cd /home/jovyan
   ```
2. Change to the `work` directory:
   ```bash
   cd work/TEAM_30_ZELTA_HPPS
   ```
3. Clone the `untrade-sdk` repository:
   ```bash
   git clone https://github.com/ztuntrade/untrade-sdk.git && cd untrade-sdk
   ```
4. Install the `untrade-sdk`:
   ```bash
   pip3 install .
   ```
5. Return to the parent directory:
   ```bash
   cd ..
   ```
6. Uninstall `numba` if installed:
   ```bash
   pip uninstall numba
   ```
7. Install all required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Running the Strategies

### For BTC Market:
Run the following command to execute the BTC strategy:
```bash
python main_1_btc.py
```

### For ETH Market:
Run the following command to execute the ETH strategy:
```bash
python main_1_eth.py
```

---

## Lunar Factor

Both strategies are identical except for the **Lunar Factor**:

- **BTC Market**: The Lunar Factor is set to **72 hours** (threshold before the new moon and full moon).
- **ETH Market**: The Lunar Factor is set to **48 hours**.

This factor determines the hours before lunar conditions are considered significant.

---

## **Note**

- Ensure all dependencies are properly installed.
- Follow the setup instructions in the specified order to avoid conflicts.

---