## Helpful commands for setting up the google cloud server

Upload docker to server
``` 
docker save dafne-server:master | ssh -C <username>@www.dafne.network docker load
``` 

Run docker on server
``` 
docker run -d -p 5000:80 --name dafne-server-job -v /mnt/data/dafne-server-db:/app/db dafne-server:master
``` 

Stop docker
```
docker stop dafne-server-job
```