"""

================================================================================

SYNTHETIC DATASET GENERATOR FOR WUSTL-EHMS-2020 TESTING

================================================================================

 

GROUND TRUTH DOCUMENTATION:

--------------------------

1. INFORMATIVE FEATURES (in "signal" mode):

   - SrcLoad: Higher on average for attacks.

   - Rate: Higher on average for attacks.

   - DIntPkt: Lower on average for attacks.

   - SrcJitter: Higher on average for attacks.

   - Temp: Shows higher variance and implausible jumps for attacks.

 

2. NOISE FEATURES:

   - All other features are independent of 'Label' with realistic ranges.

 

3. LEAKAGE / IDENTIFIER FIELDS:

   - SrcMac: '00:0a:95:9d:68:16' for normal rows, '00:14:22:01:23:45' for attack rows.

   - Attack Category: 'normal' for normal rows, 'Spoofing'/'Data Alteration' for attack rows.

   - These must be dropped prior to training to avoid leakage.

 

4. EXPECTED BEHAVIOUR:

   - "signal" mode: Models should reach ~85-95% ROC-AUC (overlap/label noise present).

   - "random" mode: Models should get ~50% ROC-AUC (unless leakage columns are not dropped,

     which would artificially inflate the score).

================================================================================

"""

 

import numpy as np

import pandas as pd

 

def generate(n_rows=16000, attack_ratio=0.125, mode="signal", seed=42):

    np.random.seed(seed)

   

    # 1. Determine labels

    n_attacks = int(n_rows * attack_ratio)

    n_normal = n_rows - n_attacks

   

    labels = np.array([0] * n_normal + [1] * n_attacks)

    np.random.shuffle(labels)

   

    # Generate base structure

    df = pd.DataFrame({'Label': labels})

   

    # 2. Constant columns

    df['Dir'] = '->'

    df['SrcAddr'] = '10.0.0.1'

    df['DstAddr'] = '10.0.0.2'

    df['Dport'] = 502

    df['SrcGap'] = 0

    df['DstGap'] = 0

    df['DIntPktAct'] = 0

    df['dMinPktSz'] = 0

    df['Trans'] = 0

    df['DstMac'] = '00:0a:95:9d:68:15'

   

    # 3. Leakage columns (directly follow ground-truth Label even in random mode)

    df['SrcMac'] = df['Label'].map({0: '00:0a:95:9d:68:16', 1: '00:14:22:01:23:45'})

    df['Attack Category'] = df['Label'].map({0: 'normal', 1: np.random.choice(['Spoofing', 'Data Alteration'], size=n_rows)})

    # Ensure mapping matches target precisely for leakage tracking

    attack_cats = np.where(df['Label'] == 0, 'normal', np.random.choice(['Spoofing', 'Data Alteration'], size=n_rows))

    df['Attack Category'] = attack_cats

   

    df['Sport'] = np.random.randint(49152, 65535, size=n_rows).astype(str)

    df['Packet_num'] = np.arange(1, n_rows + 1)

   

    # 4. Categorical feature

    df['Flgs'] = np.random.choice([' e ', ' e s ', ' e d ', ' e g '], size=n_rows, p=[0.7, 0.15, 0.1, 0.05])

   

    # 5. Non-informative network features (exponential or lognormal to match traffic skew)

    df['SrcBytes'] = np.random.lognormal(mean=6, sigma=1.5, size=n_rows).astype(int) + 1

    df['DstBytes'] = np.random.lognormal(mean=5, sigma=1.2, size=n_rows).astype(int) + 1

    df['DstLoad'] = np.random.exponential(scale=1000, size=n_rows)

    df['SIntPkt'] = np.random.exponential(scale=0.05, size=n_rows)

    df['SIntPktAct'] = np.random.exponential(scale=0.03, size=n_rows)

    df['DstJitter'] = np.random.exponential(scale=0.01, size=n_rows)

    df['sMaxPktSz'] = np.random.randint(64, 1500, size=n_rows)

    df['dMaxPktSz'] = np.random.randint(64, 1500, size=n_rows)

    df['sMinPktSz'] = np.random.randint(20, 64, size=n_rows)

    df['Dur'] = np.random.exponential(scale=2.5, size=n_rows)

    df['TotPkts'] = np.random.lognormal(mean=3, sigma=1, size=n_rows).astype(int) + 1

    df['TotBytes'] = df['SrcBytes'] + df['DstBytes']

    df['Load'] = df['DstLoad'] * 1.1

    df['Loss'] = np.random.poisson(lam=0.5, size=n_rows)

    df['pLoss'] = np.random.uniform(0.0, 0.05, size=n_rows)

    df['pSrcLoss'] = np.random.uniform(0.0, 0.03, size=n_rows)

    df['pDstLoss'] = np.random.uniform(0.0, 0.02, size=n_rows)

   

    # 6. Patient vital signs (normal baseline)

    df['Temp'] = np.random.normal(loc=37.0, scale=0.5, size=n_rows)

    df['SpO2'] = np.random.randint(95, 101, size=n_rows)

    df['Pulse_Rate'] = np.random.normal(loc=75, scale=10, size=n_rows).astype(int)

    df['SYS'] = np.random.normal(loc=120, scale=12, size=n_rows).astype(int)

    df['DIA'] = np.random.normal(loc=75, scale=8, size=n_rows).astype(int)

    df['Heart_rate'] = df['Pulse_Rate'] + np.random.randint(-2, 3, size=n_rows) # highly correlated with pulse

    df['Resp_Rate'] = np.random.normal(loc=16, scale=3, size=n_rows).astype(int)

    df['ST'] = np.random.normal(loc=0.0, scale=0.1, size=n_rows)

   

    # 7. Informative features (Signal vs. Noise dependent on mode)

    if mode == "signal":

        # Initialize default distributions

        df['SrcLoad'] = np.random.exponential(scale=500, size=n_rows)

        df['Rate'] = np.random.exponential(scale=100, size=n_rows)

        df['DIntPkt'] = np.random.exponential(scale=0.1, size=n_rows)

        df['SrcJitter'] = np.random.exponential(scale=0.05, size=n_rows)

       

        # Shift values for attack rows (Label == 1)

        attack_mask = (df['Label'] == 1)

       

        df.loc[attack_mask, 'SrcLoad'] = np.random.exponential(scale=2500, size=n_attacks)

        df.loc[attack_mask, 'Rate'] = np.random.exponential(scale=500, size=n_attacks)

        df.loc[attack_mask, 'DIntPkt'] = np.random.exponential(scale=0.01, size=n_attacks)

        df.loc[attack_mask, 'SrcJitter'] = np.random.exponential(scale=0.25, size=n_attacks)

       

        # Temp jumps for attacks

        df.loc[attack_mask, 'Temp'] = np.random.choice([35.0, 39.5, 41.0, 34.2], size=n_attacks)

       

        # Add 2% label noise (flip labels randomly)

        noise_mask = np.random.rand(n_rows) < 0.02

        df.loc[noise_mask, 'Label'] = 1 - df.loc[noise_mask, 'Label']

       

    else: # "random" mode

        df['SrcLoad'] = np.random.exponential(scale=1000, size=n_rows)

        df['Rate'] = np.random.exponential(scale=200, size=n_rows)

        df['DIntPkt'] = np.random.exponential(scale=0.05, size=n_rows)

        df['SrcJitter'] = np.random.exponential(scale=0.1, size=n_rows)

        # Temp remains normal baseline

       

    # Force appropriate column types

    cols_order = [

        'Dir', 'Flgs', 'SrcAddr', 'DstAddr', 'Sport', 'Dport', 'SrcBytes', 'DstBytes',

        'SrcLoad', 'DstLoad', 'SrcGap', 'DstGap', 'SIntPkt', 'DIntPkt', 'SIntPktAct',

        'DIntPktAct', 'SrcJitter', 'DstJitter', 'sMaxPktSz', 'dMaxPktSz', 'sMinPktSz',

        'dMinPktSz', 'Dur', 'Trans', 'TotPkts', 'TotBytes', 'Load', 'Loss', 'pLoss',

        'pSrcLoss', 'pDstLoss', 'Rate', 'SrcMac', 'DstMac', 'Packet_num', 'Temp',

        'SpO2', 'Pulse_Rate', 'SYS', 'DIA', 'Heart_rate', 'Resp_Rate', 'ST',

        'Attack Category', 'Label'

    ]

   

    return df[cols_order]

 

# --- Generation & Sanity Checks ---

for mode in ["signal"]:

    print(f"\nGenerating synthetic dataset in '{mode}' mode...")

    df_syn = generate(mode=mode)

   

    # Save to disk

    filename = f"synthetic_{mode}.csv"

    df_syn.to_csv(filename, index=False)

    print(f"Saved to {filename}")

   

    # Sanity verification

    print(f"Shape: {df_syn.shape}")

    print(f"Label distribution:\n{df_syn['Label'].value_counts(normalize=True).to_string()}")

   

    # Unique counts to locate constants

    unique_counts = df_syn.nunique()

    constants = unique_counts[unique_counts == 1].index.tolist()

    print(f"Constant columns detected (should be 10): {len(constants)} -> {constants}")

   

    # Verify informative feature means

    print("Informative features average values by Class:")

    informative_cols = ['SrcLoad', 'Rate', 'DIntPkt', 'SrcJitter', 'Temp']

    print(df_syn.groupby('Label')[informative_cols].mean())