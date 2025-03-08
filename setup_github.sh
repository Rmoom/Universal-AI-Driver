#!/bin/bash

# Warna untuk output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

echo -e "${YELLOW}Setup GitHub Repository untuk Universal AI Driver${NC}"

# Minta username GitHub
read -p "Masukkan username GitHub anda: " github_username

# Minta personal access token
read -sp "Masukkan GitHub personal access token anda: " github_token
echo

# Nama repo
repo_name="universal-ai-driver"

echo -e "\n${GREEN}Membuat repository di GitHub...${NC}"

# Buat repo di GitHub menggunakan API
curl -H "Authorization: token $github_token" \
     -d "{\"name\":\"$repo_name\",\"description\":\"Universal AI Driver untuk Linux Silverblue dengan Hybrid GPU Processing\",\"private\":false,\"auto_init\":true,\"gitignore_template\":\"Python\",\"license_template\":\"mit\"}" \
     https://api.github.com/user/repos

echo -e "\n${GREEN}Clone repository...${NC}"
git clone https://github.com/$github_username/$repo_name.git
cd $repo_name

echo -e "\n${GREEN}Menyalin fail-fail projek...${NC}"
# Salin semua fail dan folder
cp -r ../src .
cp -r ../config .
cp -r ../systemd .
cp ../requirements.txt .
cp ../README.md .
cp ../LICENSE .
cp ../.gitignore .

echo -e "\n${GREEN}Commit dan push ke GitHub...${NC}"
git add .
git config user.email "your-email@example.com"
git config user.name "Your Name"
git commit -m "Initial commit: Universal AI Driver"
git push https://$github_token@github.com/$github_username/$repo_name.git main

echo -e "\n${GREEN}Selesai! Repository anda boleh dilihat di:${NC}"
echo "https://github.com/$github_username/$repo_name" 