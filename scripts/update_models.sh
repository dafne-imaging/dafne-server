#!/bin/bash
set -e  # stop on error 

echo "Backup old models..."
rm -rf ~/dev/dafne/models_OLD
mv ~/dev/dafne/models ~/dev/dafne/models_OLD
mkdir ~/dev/dafne/models

echo "Generating models..."
cd  ~/dev/dafne
python generate_thigh_split_model.py  
python generate_leg_split_model.py  
cd ~

echo "Backup old models..."
rm -rf ~/dev/dafne-server/db/models_OLD
mv ~/dev/dafne-server/db/models ~/dev/dafne-server/db/models_OLD
mkdir ~/dev/dafne-server/db/models

echo "Copy to server repo..."
mkdir -p ~/dev/dafne-server/db/models/Leg/uploads
mkdir -p ~/dev/dafne-server/db/models/Thigh/uploads
cp ~/dev/dafne/models/Leg_1610001000.model ~/dev/dafne-server/db/models/Leg/1610001000.model
cp ~/dev/dafne/models/Thigh_1610001000.model ~/dev/dafne-server/db/models/Thigh/1610001000.model

# echo "Uploading to server..."
# scp -r ~/dev/dafne-server/db/models j_wasserthal_gmx_de@www.dafne.network:/mnt/data/dafne-server-db

# echo "Testing..."
# cd ~/dev/dafne-script
# python test_api.py
