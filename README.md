Install system dependencies:
```
sudo apt-get update
sudo apt-get install build-essential python3 python3-dev python3-opencv python3-pip python3-venv libagg-dev libpotrace-dev pkg-config
```

Create a virtual environment and install packages:
```
python3 -m venv bezier-env
source bezier-env/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```
