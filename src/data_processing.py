import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import RandomOverSampler

def to_numeric_clean(x):
    if pd.isna(x):
        return np.nan

    s = str(x).strip()

    # # if the string contains a comma, treat it as invalid and return NaN
    if "," in s:
        return np.nan
    # Fix the issue with values like "0x00000000,0x00004000"
    if s == "" or s == "-":
        return np.nan

    return pd.to_numeric(s, errors="coerce")

def hex_to_int(x):
    if pd.isna(x):
        return np.nan
    s = str(x).strip()
    # Fix lỗi: "0x00000000,0x00004000"
    if "," in s:
        s = s.split(",")[0]
    if s.startswith("0x"):
        try:
            return int(s, 16)
        except:
            return np.nan
    return pd.to_numeric(s, errors="coerce")


dfs = []
arr1 = [1,2,3,4,5,6]
for i in arr1:
    df = pd.read_csv(f"data/arp_spoofing_{i}_attack.tsv", sep="\t")
    df["label"] = 1
    print(df.shape)
    dfs.append(df)

df = pd.read_csv(f"data/arp_spoofing_1_benign.tsv", sep="\t")
df["label"] = 0
print(f"Shape {df.shape}")
dfs.append(df)
combined = pd.concat(dfs, ignore_index=True, sort=False)
print("Label distribution:")
print(combined["label"].value_counts(dropna=False))
combined.to_csv("data/arp_spoofing_all_combined.csv", index=False)

df = pd.read_csv("data/arp_spoofing_all_combined.csv")

# Convert ["ip.proto", "ip.len", "ip.ttl", "ip.hdr_len"] to numeric
cols_ip = ["ip.proto", "ip.len", "ip.ttl", "ip.hdr_len"]
for col in cols_ip:
    df[col] = df[col].apply(to_numeric_clean)


# Convert (ip.flags + tcp.flags) to  decimal
df["ip.flags"] = df["ip.flags"].apply(hex_to_int)
df["tcp.flags"] = df["tcp.flags"].apply(hex_to_int)


# Binary_Encode ARP & ICMP (0/1)
df["arp"] = df["arp"].apply(lambda x: 1 if pd.notna(x) else 0)
df["icmp"] = df["icmp"].apply(lambda x: 1 if pd.notna(x) else 0)


# Disable TCP/UDP fields
tcp_features = [
    'tcp.srcport', 'tcp.dstport', 'tcp.flags', 'tcp.flags.syn',
    'tcp.flags.ack', 'tcp.flags.reset', 'tcp.window_size',
    'tcp.checksum.status', 'tcp.len', 'tcp.urgent_pointer', 'tcp.time_delta'
]
udp_features = ['udp.srcport', 'udp.dstport']

df.loc[df["ip.proto"] == 6, udp_features] = -1
df.loc[df["ip.proto"] == 17, tcp_features] = -1

# Convert all remaining object-type columns to numeric 
for col in df.select_dtypes(include=['object']).columns:
    df[col] = df[col].apply(to_numeric_clean)


# Fill NaN
df = df.fillna(-1)


# Split features (X) and labels (y)
y = df["label"]
X = df.drop(columns=["label"])

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)


# Oversample TRAIN
ros = RandomOverSampler(random_state=42)
X_train_res, y_train_res = ros.fit_resample(X_train, y_train)

# MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train_res), columns=X_train_res.columns)
X_test_scaled  = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)


# SAVE
train_df = pd.concat([X_train_scaled, y_train_res.reset_index(drop=True)], axis=1)
test_df  = pd.concat([X_test_scaled,  y_test.reset_index(drop=True)], axis=1)

train_df.to_csv("arp_train_preprocessed.csv", index=False)
test_df.to_csv("arp_test_preprocessed.csv", index=False)


# Final Check
print("Train NaN:", train_df.isna().sum().sum())
print("Test  NaN:", test_df.isna().sum().sum())
print("Train Min:", train_df.min().min(), "Max:", train_df.max().max())
print("Test  Min:", test_df.min().min(), "Max:", test_df.max().max())
print(y_train_res.value_counts())
