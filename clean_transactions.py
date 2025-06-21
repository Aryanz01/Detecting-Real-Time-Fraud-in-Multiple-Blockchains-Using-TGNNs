import pandas as pd

def clean_transaction_data(file_path, output_path):
    # Read CSV
    df = pd.read_csv(file_path)

    # Rename columns for consistency (remove spaces and lowercase)
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    # Convert timestamp to datetime if not already
    if df['timestamp'].dtype != 'datetime64[ns]':
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s', errors='coerce')

    # Convert value fields to numeric
    df['amount'] = pd.to_numeric(df['amount'], errors='coerce')
    df['gas_fee'] = pd.to_numeric(df['gas_fee'], errors='coerce')

    # Standardize blockchain type to lowercase
    df['blockchain_type'] = df['blockchain_type'].str.lower()

    # Fill missing smart contract addresses with "N/A"
    df['smart_contract_address'] = df['smart_contract_address'].fillna("N/A")

    # Optional: drop rows with critical missing data
    df.dropna(subset=['transaction_hash', 'sender_address', 'receiver_address', 'timestamp'], inplace=True)

    # Save cleaned data
    df.to_csv(output_path, index=False)
    print(f"Processed data saved to: {output_path}")

if __name__ == "__main__":
    input_file = "raw_transactions.csv"
    output_file = "processed_transactions.csv"
    clean_transaction_data(input_file, output_file)