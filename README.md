# MAST Rhythm similarity
A python project to compare distance functions for calculation of similarities between extracted rhythm files.
Datasets include jury scores along with time intervals of rhythms for records.
SparkMLLib and sklearn models are used to create machine learning prediction models.

## Installing Essentia
In linux, simply run
```
sudo apt-get -y install build-essential libyaml-dev libfftw3-dev libavcodec-dev libavformat-dev libavutil-dev libavresample-dev python-dev libsamplerate0-dev libtag1-dev
sudo apt-get -y install python-numpy-dev python-numpy python-yaml
git clone https://github.com/MTG/essentia.git
cd essentia
sudo ./waf configure --mode=release --build-static --with-python --with-cpptests --with-examples --with-vamp
sudo ./waf
sudo ./waf install
```

## Dependencies
```
sklearn
numpy
scipy
essentia
python >= 2.7
```

## Sample datasets
Test datasets will be provided.

## Contributing
If you find any bug, open an issue or just PR.

## Authors
* **Zafer Çavdar** - [zafercavdar](https://github.com/zafercavdar)
* **Burak Uyar** - [burakuyar](https://github.com/burakuyar)


