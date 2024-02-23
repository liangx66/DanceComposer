# DanceComposer

## 1. Prerequisite

The environment prerequisites are as follows:

- python 3.8

## 2. Dataset

The datasets utilized in our paper are as follows:

### 2.1 AIST

**AIST Dance Video Database (AIST Dance DB)** is a shared database containing original street dance videos with copyright-cleared dance music. The database is available [here](https://aistdancedb.ongaaccel.jp/).

### 2.2 GTZAN

**The GTZAN dataset** is a collection of 1,000 audio files spanning 10 music genres, all having a length of 30 seconds. The audio files are available [here](https://www.kaggle.com/datasets/andradaolteanu/gtzan-dataset-music-genre-classification).

### 2.3 GrooveMIDI

The **Groove MIDI Dataset (GMD)** is composed of 13.6 hours of aligned MIDI and (synthesized) audio of human-performed, tempo-aligned expressive drumming.  The MIDI data is available in the [documentation](https://magenta.tensorflow.org/datasets/groove).

### 2.4 LPD

The **Lakh Pianoroll Dataset (LPD)** is a collection of 174,154 multitrack pianorolls derived from the Lakh MIDI Dataset (LMD). We use its subset [lpd-5-cleansed](https://drive.google.com/uc?id=1yz0Ma-6cWTl6mhkrLnAVJ7RNzlQRypQ5) that contains 21,425 five-track pianorolls.

## 3. Preparation

- extract human skeleton keypoints using OpenPose
- extract ground truth music beats
- extract log mel-scaled spectrogram
- convert drum track/multi-track MIDI into token sequence

## 4. Training

### 4.1 MBPN

To train the MBPN.

```python
python ./src/MBPN/train_MBPN.py
```

### 4.2 SSM

To pre-train the Dance style embedding network on AIST.

```python
python ./src/SSM/train_dance_network.py
```

To pre-train the Music style embedding network on GTZAN.

```python
python ./src/SSM/train_music_network.py
```

To jointly train the Dance and music style embedding networks.

```python
python ./src/SSM/train_joint.py
```

### 4.3 PCMG

To train the Drum Transformer on GrooveMIDI.

```python
python ./src/PCMG/train_drum_Transformer.py
```

To train the Multi-track Transformer on LPD.
```python
python ./src/PCMG/train_multi_Transformer.py
```


