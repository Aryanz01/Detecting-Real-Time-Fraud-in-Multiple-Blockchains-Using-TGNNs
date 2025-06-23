# multi_chain_stream.py

import json
import time
from decimal import Decimal
from web3 import Web3
import requests
import pandas as pd
from data_preprocessing.clean_transactions import clean_transaction_data

# ------------------------- CONFIG ------------------------- #
ETHEREUM_RPC = "https://mainnet.infura.io/v3/65aeaa5ff07340e38fc51789e05391a5"
GETBLOCK_SOLANA_RPC = "https://go.getblock.io/5f0c859f62284e51903f6d63321764e1"
BITCOIN_API_URL = "https://blockstream.info/api"

PARAMETERS = {
    "transaction_hash": True,
    "sender_address": True,
    "receiver_address": True,
    "timestamp": True,
    "amount": True,
    "token_type": True,
    "gas_fee": True,
    "blockchain_type": True,
    "smart_contract_address": True,
}

# ------------------------- ETHEREUM ------------------------- #
def get_ethereum_blocks(num_blocks):
    w3 = Web3(Web3.HTTPProvider(ETHEREUM_RPC))
    latest_block = w3.eth.block_number
    blocks = []

    for block_num in range(latest_block, latest_block - num_blocks, -1):
        block = w3.eth.get_block(block_num, full_transactions=True)
        for tx in block.transactions:
            tx_data = extract_tx_data(tx, "Ethereum", block.timestamp)
            blocks.append(tx_data)
    return blocks

# ------------------------- SOLANA ------------------------- #
def get_solana_transactions(limit):
    payload_sig = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "getSignaturesForAddress",
        "params": ["Vote111111111111111111111111111111111111111", {"limit": limit}]
    }
    res_sig = requests.post(GETBLOCK_SOLANA_RPC, json=payload_sig)
    if res_sig.status_code != 200:
        print("Error fetching signatures:", res_sig.text)
        return []

    sig_list = res_sig.json().get("result", [])
    print("Fetched signatures:", sig_list)  # 🛠 Debug

    transactions = []
    for sig_obj in sig_list:
        sig = sig_obj.get("signature")
        if not sig:
            continue

        payload_tx = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "getTransaction",
            "params": [sig, {"encoding": "json", "commitment": "confirmed"}]
        }
        tx_res = requests.post(GETBLOCK_SOLANA_RPC, json=payload_tx)
        if tx_res.status_code != 200:
            print(f"Error fetching Tx {sig}:", tx_res.text)
            continue

        tx_data = tx_res.json().get("result")
        if tx_data:
            transactions.append(extract_tx_data_solana(tx_data, "Solana"))

    print(f"Retrieved {len(transactions)} Solana txns")  # 🛠 Debug
    return transactions

# ------------------------- BITCOIN ------------------------- #
def get_bitcoin_blocks(num_blocks):
    latest_block_hash = requests.get(f"{BITCOIN_API_URL}/blocks/tip/hash").text
    blocks = []

    for _ in range(num_blocks):
        block_data = requests.get(f"{BITCOIN_API_URL}/block/{latest_block_hash}").json()
        for txid in block_data.get("tx", []):
            tx_data = extract_tx_data_bitcoin(txid, block_data, "Bitcoin")
            if tx_data:  # Skip None entries
                blocks.append(tx_data)
        latest_block_hash = block_data.get("previousblockhash")
    return blocks

def get_bitcoin_transaction_details(txid):
    url = f"{BITCOIN_API_URL}/tx/{txid}"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# ------------------------- EXTRACTORS ------------------------- #
def extract_tx_data(tx, blockchain, timestamp):
    return {
        "Transaction Hash": tx.hash.hex() if PARAMETERS["transaction_hash"] else None,
        "Sender Address": tx["from"] if PARAMETERS["sender_address"] else None,
        "Receiver Address": tx.to if PARAMETERS["receiver_address"] else None,
        "Timestamp": timestamp if PARAMETERS["timestamp"] else None,
        "Amount": float(Web3.from_wei(tx.value, 'ether')) if PARAMETERS["amount"] else None,
        "Token Type": "ETH" if PARAMETERS["token_type"] else None,
        "Gas Fee": float(Web3.from_wei(tx.gas * (tx.gasPrice or 0), 'ether')) if PARAMETERS["gas_fee"] else None,
        "Blockchain Type": blockchain if PARAMETERS["blockchain_type"] else None,
        "Smart Contract Address": tx.to if tx.to and tx.input != '0x' and PARAMETERS["smart_contract_address"] else None,
    }

def extract_tx_data_solana(tx, blockchain):
    meta = tx.get("meta", {})
    transaction = tx.get("transaction", {})
    message = transaction.get("message", {})
    account_keys = message.get("accountKeys", [])

    return {
        "Transaction Hash": transaction.get("signatures", [None])[0] if PARAMETERS["transaction_hash"] else None,
        "Sender Address": account_keys[0] if PARAMETERS["sender_address"] and len(account_keys) > 0 else None,
        "Receiver Address": account_keys[1] if PARAMETERS["receiver_address"] and len(account_keys) > 1 else None,
        "Timestamp": tx.get("blockTime") if PARAMETERS["timestamp"] else None,
        "Amount": meta.get("postBalances", [None])[0] if PARAMETERS["amount"] else None,
        "Token Type": "SOL" if PARAMETERS["token_type"] else None,
        "Gas Fee": meta.get("fee") if PARAMETERS["gas_fee"] else None,
        "Blockchain Type": blockchain if PARAMETERS["blockchain_type"] else None,
        "Smart Contract Address": None,
    }

def extract_tx_data_bitcoin(txid, block_data, blockchain):
    tx_details = get_bitcoin_transaction_details(txid)
    if not tx_details:
        return None

    inputs = tx_details.get("vin", [])
    outputs = tx_details.get("vout", [])

    sender_address = inputs[0].get("prevout", {}).get("scriptpubkey_address") if inputs else None
    receiver_address = outputs[0].get("scriptpubkey_address") if outputs else None
    amount = outputs[0].get("value") / 1e8 if outputs else None  # Convert satoshis to BTC

    return {
        "Transaction Hash": txid if PARAMETERS["transaction_hash"] else None,
        "Sender Address": sender_address if PARAMETERS["sender_address"] else None,
        "Receiver Address": receiver_address if PARAMETERS["receiver_address"] else None,
        "Timestamp": block_data.get("timestamp") if PARAMETERS["timestamp"] else None,
        "Amount": amount if PARAMETERS["amount"] else None,
        "Token Type": "BTC" if PARAMETERS["token_type"] else None,
        "Gas Fee": None,  # You can fetch fee if required
        "Blockchain Type": blockchain if PARAMETERS["blockchain_type"] else None,
        "Smart Contract Address": None,
    }

# ------------------------- MAIN ------------------------- #
def decimal_default(obj):
    if isinstance(obj, Decimal):
        return float(obj)
    return str(obj)

def stream_blocks(eth_blocks=1, solana_txs=1, btc_blocks=1, output_file="raw_transactions.csv"):
    eth_data = get_ethereum_blocks(eth_blocks)
    sol_data = get_solana_transactions(solana_txs)
    btc_data = get_bitcoin_blocks(btc_blocks)

    all_data = eth_data + sol_data + btc_data

    if not all_data:
        print("No data to save.")
        return

    # Convert to DataFrame
    df = pd.DataFrame(all_data)

    # Save to CSV
    df.to_csv(output_file, index=False)
    print(f"Saved raw transaction data to {output_file}")

if __name__ == "__main__":
    raw_file = "raw_transactions.csv"
    processed_file = "processed_transactions.csv"

    stream_blocks(eth_blocks=1, solana_txs=1, btc_blocks=1, output_file=raw_file)

    # Run preprocessing after extraction
    clean_transaction_data(raw_file, processed_file)
