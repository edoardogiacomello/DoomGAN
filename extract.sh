PROJECT_HOME=/home/Doom/DoomPCGML
echo "Extracting the dataset..."
unzip $PROJECT_HOME/dataset/128x128-one-floor.zip -d $PROJECT_HOME/dataset/
echo "Extracting the netowrk with feaures"
unzip $PROJECT_HOME/models/with_features.zip -d $PROJECT_HOME/artifacts/
