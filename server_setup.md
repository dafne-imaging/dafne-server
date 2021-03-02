## Helpful commands for setting up the google cloud server


Updating code on server:
``` 
cd /mnt/data/code/dafne-server
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

Backup to local harddrive
``` 
rsync -avz <username>@www.dafne.network:/mnt/data/dafne-server-db /mnt/jay_hdd/backup
``` 
