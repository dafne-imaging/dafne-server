## Helpful commands for setting up the google cloud server

Regenerating original models (inside of client code):

1. Put original weights into `dafne/weights`
2. Generate models:
```
python generate_thigh_model.py  
python generate_leg_model.py  
python generate_thigh_split_model.py  
python generate_leg_split_model.py  
python generate_classifier.py  
``` 
3. Copy to `dafne-server/db/models`
`scp -r ~/dev/dafne/models <username>@www.dafne.network:/mnt/data/dafne-server-db`


Updating code on server:
``` 
cd dev/dafne-server
git pull
cd dl
git pull
cd ..
docker build -t dafne-server:master .
``` 



Run docker on server
``` 
docker run -d --restart always -p 5000:80 --name dafne-server-job -v /mnt/data/dafne-server-db:/app/db dafne-server:master
``` 

Stop docker
```
docker stop dafne-server-job
```


## Other commands

Upload docker to server
``` 
docker save dafne-server:master | ssh -C <username>@www.dafne.network docker load
``` 
