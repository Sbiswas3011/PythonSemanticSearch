Basic Commands: to load local files into server

$KEY = "$env:USERPROFILE\Documents\SS_Instance_medium.pem"
$IP = "YOUR_NEW_EC2_PUBLIC_IP"

scp -i $KEY -r `"C:\Users\JAYANTA\Desktop\LLM_Work\models\embeddinggemma2" `
"ubuntu@${IP}:/opt/semantic-search/model/"

scp -i $KEY `
  "C:\Users\JAYANTA\Desktop\PythonCases\embeddings.npz" `
  "ubuntu@${IP}:/opt/semantic-search/PythonSemanticSearch/"

scp -i $KEY `
  "C:\Users\JAYANTA\Desktop\PythonCases\convert_results.jsonl" `
  "ubuntu@${IP}:/opt/semantic-search/PythonSemanticSearch/"

ssh -i ${KEY} ubuntu@${IP}

Basic Commands: to setup server on startup

# Update package lists
sudo apt update

# Install basic utilities
sudo apt install -y git curl ca-certificates

# Install Docker and the Compose plugin
sudo apt install -y docker.io docker-compose-v2

# Enable Docker at boot and start it now
sudo systemctl enable --now docker

# Verify installations
git --version
sudo docker --version
sudo docker compose version

# Test Docker
sudo docker run hello-world

sudo mkdir -p /opt/semantic-search
sudo chown -R "$USER":"$USER" /opt/semantic-search

cd /opt/semantic-search

# Create directories for the model
mkdir -p model

# Clone your GitHub repository
git clone https://github.com/Sbiswas3011/PythonSemanticSearch.git

cd PythonSemanticSearch

sudo docker compose logs -f
sudo docker compose ps
VITE_API_BASE=https://your-api.example.com npm run build