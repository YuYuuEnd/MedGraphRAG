#!/bin/bash

# Navigate to the project directory
cd e:/Yuu/an/YuE/RAG

# Add a README file (optional)
echo "# YuE" > README.md
git config --global user.name "Vu Van An"
git config --global user.email "rem7ram@gmail.com"

# Initialize the repository (if not already initialized)
git init

# Add all files in the directory to the staging area
git add .

# Commit the changes with a meaningful message
git commit -m "Initial commit of RAG project"

# Set the branch to main
git branch -M main

# Add or update the remote repository
git remote add origin https://github.com/YuYuuEnd/MedGraphRAG.git

# Push the changes to GitHub
git push -u origin main