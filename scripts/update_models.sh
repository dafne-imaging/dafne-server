#!/bin/bash
set -e  # stop on error 

echo "Backup old models..."
rm -rf ~/dev/dafne/models_OLD
mv ~/dev/dafne/models ~/dev/dafne/models_OLD
mkdir ~/dev/dafne/models

echo "Generating models..."
cd  ~/dev/dafne
python generate_thigh_model.py  
python generate_leg_model.py  
python generate_thigh_split_model.py  
python generate_leg_split_model.py  
python generate_classifier.py  
cd ~

echo "Backup old models..."
rm -rf ~/dev/dafne-server/db/models_OLD
mv ~/dev/dafne-server/db/models ~/dev/dafne-server/db/models_OLD
mkdir ~/dev/dafne-server/db/models

echo "Copy to server repo..."
mv ~/dev/dafne/models/Classifier_1603281030.model ~/dev/dafne-server/db/models/Classifier/1603281030.model
mv ~/dev/dafne/models/Leg_1603281013.model ~/dev/dafne-server/db/models/Classifier/1603281013.model
mv ~/dev/dafne/models/Leg-Split_1603281013.model ~/dev/dafne-server/db/models/Classifier/1603281013.model
mv ~/dev/dafne/models/Thigh_1603281020.model ~/dev/dafne-server/db/models/Classifier/1603281020.model
mv ~/dev/dafne/models/Thigh-Split_1603281020.model ~/dev/dafne-server/db/models/Classifier/1603281020.model

echo "Uploading to server..."
scp -r ~/dev/dafne-server/db/models j_wasserthal_gmx_de@www.dafne.network:/mnt/data/dafne-server-db

echo "Testing..."
cd ~/dev/dafne-script
python test_api.py
