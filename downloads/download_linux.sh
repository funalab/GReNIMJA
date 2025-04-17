#!/bin/bash

# Download and extract datasets
wget -O datasets.tar.gz "https://drive.usercontent.google.com/download?id=15sZ_47YSZQQLMTAloV49zRrFGTXwxaA1&confirm=xxx"
tar zxvf datasets.tar.gz

# Download and extract embeddings
wget -O embeddings.tar.gz "https://drive.usercontent.google.com/download?id=1nU3sJSujcZ1rffKRpw0K5OSVH3uPvJLa&confirm=xxx"
tar zxvf embeddings.tar.gz

# Download and extract learned model
wget -O models.tar.gz "https://drive.usercontent.google.com/download?id=1ohzA4q9qMtvSKlbz4hgR9HKoK_TzcN7n&confirm=xxx"
tar zxvf models.tar.gz