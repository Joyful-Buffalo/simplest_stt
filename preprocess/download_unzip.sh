set -e
ROOT=dataset/aishell1
sudo mkdir -p "$ROOT"
sudo chown -R "$USER":"$USER" "$ROOT"
cd "$ROOT"
if [ ! -f "data_aishell.tgz" ]; then
    wget https://www.openslr.org/resources/33/data_aishell.tgz
fi
tar -zxvf data_aishell.tgz
cd data_aishell/wav
find . -name '*.tar.gz' -execdir sudo tar -zxvf {} \;
find . -name '*.tar.gz' -delete
cd ../../
if [ ! -f "resource_aishell.tgz" ]; then
    wget https://www.openslr.org/resources/33/resource_aishell.tgz
fi
tar -zxvf resource_aishell.tgz