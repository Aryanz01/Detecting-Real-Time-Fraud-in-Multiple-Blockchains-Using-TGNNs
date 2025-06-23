# import pandas as pd
# import networkx as nx
# import matplotlib.pyplot as plt

# # Load CSV instead of JSON
# df = pd.read_csv(
#     "processed_transactions.csv",
#     dtype={
#         'Transaction Hash': str,
#         'Sender Address': str,
#         'Receiver Address': str,
#         'Timestamp': str,         # use str to avoid any casting issues
#         'Amount': str,            # use str to handle both 0 and float strings
#         'Token Type': str,
#         'Gas Fee': str,           # or float if all are valid numbers
#         'Blockchain Type': str,
#         'Smart Contract Address': str
#     },
#     low_memory=False
# )

# # Create directed graph
# G = nx.DiGraph()

# # Add edges using sender and receiver
# for _, row in df.iterrows():
#     sender = row['Sender Address']
#     receiver = row['Receiver Address']
#     if pd.notna(sender) and pd.notna(receiver):
#         G.add_edge(sender, receiver)

# # Draw
# plt.figure(figsize=(12, 8))
# nx.draw(G, with_labels=True, node_size=50, font_size=5)
# plt.show()

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import torch
from torch_geometric.data import TemporalData
import os

def build_temporal_graph(csv_file, save_path="graphs/temporal_graph.pt"):
    # Load processed data
    df = pd.read_csv(csv_file)
    
    # Drop rows with missing critical fields
    df = df.dropna(subset=['sender_address', 'receiver_address', 'timestamp', 'amount'])

    # Encode wallet addresses to integer node IDs
    addr_encoder = LabelEncoder()
    all_addresses = pd.concat([df['sender_address'], df['receiver_address']])
    addr_encoder.fit(all_addresses)

    df['sender_id'] = addr_encoder.transform(df['sender_address'])
    df['receiver_id'] = addr_encoder.transform(df['receiver_address'])

    # Convert timestamp to UNIX format (seconds) - no warnings in Pandas 2.0+
    df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
    df['unix_ts'] = df['timestamp'].values.astype(np.int64) // 10**9

    # Sort by timestamp to ensure correct temporal sequence
    df = df.sort_values(by='unix_ts').reset_index(drop=True)

    # Normalize edge features: amount, gas_fee
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0.0)
    df['gas_fee'] = pd.to_numeric(df['gas_fee'], errors='coerce').fillna(0.0)

    # Encode token type and blockchain type as categorical
    token_encoder = LabelEncoder()
    chain_encoder = LabelEncoder()

    df['token_type_enc'] = token_encoder.fit_transform(df['token_type'].fillna("unknown"))
    df['blockchain_type_enc'] = chain_encoder.fit_transform(df['blockchain_type'].fillna("unknown"))

    # Binary flag for smart contract interaction
    df['has_contract'] = df['smart_contract_address'].apply(lambda x: 0 if x == "N/A" or pd.isna(x) else 1)

    # Combine edge features into one matrix
    edge_features = torch.tensor(df[['amount', 'gas_fee', 'token_type_enc', 'blockchain_type_enc', 'has_contract']].values, dtype=torch.float)

    # Prepare temporal graph data
    edge_index = torch.tensor([df['sender_id'].values, df['receiver_id'].values], dtype=torch.long)
    edge_timestamp = torch.tensor(df['unix_ts'].values, dtype=torch.long)

    # Build the PyTorch Geometric Temporal object
    data = TemporalData(
        edge_index=edge_index,
        edge_attr=edge_features,
        t=edge_timestamp
    )

    # Save the graph for reuse
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(data, save_path)
    print(f"Temporal graph saved to {save_path}")

    return data, addr_encoder

# Usage
if __name__ == "__main__":
    data, encoder = build_temporal_graph("processed_transactions.csv")
    print("✅ Temporal Graph Created Successfully!")
    print(f"Number of edges: {data.num_edges}")
    print(f"Edge index shape: {data.edge_index.shape}")
    print(f"Edge feature shape: {data.edge_attr.shape}")



    import torch
import pandas as pd

# Load your saved temporal graph
data = torch.load('graphs/temporal_graph.pt')
# Convert edge index and edge attributes to DataFrame
edge_index = data.edge_index.numpy()
edge_attr = data.edge_attr.numpy()
timestamps = data.t.numpy()

# Build the dataframe
df = pd.DataFrame({
    'sender_id': edge_index[0],
    'receiver_id': edge_index[1],
    'amount': edge_attr[:, 0],
    'gas_fee': edge_attr[:, 1],
    'token_type': edge_attr[:, 2],
    'blockchain_type': edge_attr[:, 3],
    'has_contract': edge_attr[:, 4],
    'timestamp': timestamps
})

# Save to CSV
df.to_csv('graph_edges.csv', index=False)
print('✅ Graph edges exported to graph_edges.csv')

