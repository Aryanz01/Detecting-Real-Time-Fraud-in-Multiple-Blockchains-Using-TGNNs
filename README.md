⚙️ Setup Instructions

🛠️ How to Run:

Clone the Repository:
git clone https://github.com/Aryanz01/Detecting-Real-Time-Fraud-in-Multiple-Blockchains-Using-TGNNs.git
cd Detecting-Real-Time-Fraud-in-Multiple-Blockchains-Using-TGNNs

Install basic dependencies:
pip install -r requirements.txt

Run Setup Script to Install Dependencies:
bash setup.sh

This will:
Create a virtual environment venv
Install all base dependencies
Install a compatible PyTorch version for Apple Silicon
Install all PyTorch Geometric libraries


(Alternative) 
Manual Installation:
If you don't want to use the setup script:
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
pip install torch==2.2.0 torchvision torchaudio
pip install torch-scatter torch-sparse torch-cluster torch-spline-conv torch-geometric -f https://data.pyg.org/whl/torch-2.2.0+cpu.html
pip install torch-geometric-temporal


🏃 Running the Code

Extract Blockchain Transactions:
python block_ether.py

Preprocess Data and Build Temporal Graph:
python data_preprocessing/graph_bilder.py

Output Files:
Cleaned transactions: processed_transactions.csv
Temporal graph: graphs/temporal_graph.pt
Graph edges: graph_edges.csv
