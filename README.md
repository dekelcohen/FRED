# FRED: The Florence RGB-Event Drone Dataset


<div align="center">
  <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/sliders_auto/montage.webp?raw=true" 
     alt="Demo HDR"
    width=720/>
</div>


Official repository for the  **FRED dataset**, a large-scale multimodal dataset specifically designed for drone detection, tracking, and trajectory forecasting, with spatiotemprally  **synchronized RGB and event data**.
It includes **train** and **test** splits with zipped subfolders for each sequence.

The dataset can also be **downloaded** from [here](https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp?usp=share_link).
The dataset splits in .txt format, along with the alternative **challenging split**, can be found [here](https://github.com/miccunifi/FRED/tree/main/dataset_splits).

**Demos and examples** can be found in the official [website](https://miccunifi.github.io/FRED/). Check it out, it's pretty cool! :)


---

## 📂 Dataset Structure

```
FRED/
 ├── train/
 │    ├── 0.zip
 │    ├── 1.zip
 │    └── ...
 ├── test/
 │    ├── 100.zip
 │    ├── 101.zip
 │    └── ...
```

Each `.zip` file corresponds to one sequence (rgb frames, event data, and annotations).
Event data comprends both already extracted event frames and the relative **.hdf5 file**, containing the raw event stream.

---

## 📝 Annotation Format

Each sequence includes two **`.txt` annotation file** with bounding box and identity information for every frame.  
Since rgb images are padded to enable a same coordinate space between the two modalities, the event videos have additional boxes corresponding to the padded area in the RGB.
To better divide the 2 cases, we divided the annotation into **`coordinates.txt`** that represents the **extended boxes** and the **`coordinates_rgb.txt`** for the boxes **excluding the padding**.

We reccomend using the **extended boxes** version since it facilitates the training and the overall cases in which the drone falls into the padding is relatively limited when compared to the number of samples.

The format of the annotations is:

```
time: x1, y1, x2, y2, id, class
```
- **time**  → time relative to the start of the recording in 'seconds.microseconds' for the annotation
- **x1, y1** → top-left corner of the bounding box  
- **x2, y2** → bottom-right corner of the bounding box  
- **id** → unique identifier for the drone, consistent across frames (for tracking)  
- **class** → drone type label  

📌 Example:
```
1.33332: 490.0, 413.0, 539.0, 448.0, 1, DJI Mini 2
6.33327: 609.0, 280.0, 651.0, 308.0, 2, DarwinFPV cineape20 
```

This structure is compatible with standard detection and tracking pipelines, while maintaining instance-level identity across time.

---

## 📥 Download


### Clone the entire dataset
```bash
git lfs install
git clone https://huggingface.co/datasets/GabrieleMagrini/FRED
```

### Download specific sequences
```bash
wget https://huggingface.co/datasets/GabrieleMagrini/FRED/train/0.zip
wget https://huggingface.co/datasets/GabrieleMagrini/FRED/test/100.zip
```

### Use with 🤗 Datasets
```python
from datasets import load_dataset

# Load full dataset
ds = load_dataset("GabrieleMagrini/FRED")

# Load specific split
train_set = load_dataset("GabrieleMagrini/FRED", split="train")
test_set  = load_dataset("GabrieleMagrini/FRED", split="test")
```
### Using Drive

The dataset can be downloaded from [here](https://drive.google.com/drive/folders/1pISIErXOx76xmCqkwhS3-azWOMlTKZMp?usp=share_link).

The dataset splits can be found [here](https://github.com/miccunifi/FRED/tree/main/dataset_splits).

Demos and examples can be found in the official [website](https://miccunifi.github.io/FRED/)

---

## 🖼️ Examples

<div align="center">
            <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #fff;">Night</h1>
</div>
<div style="text-align:center;">
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/rgb_night.gif?raw=true" width="400"/>
  </div>
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/event_night.gif?raw=true" width="400"/>
  </div>
</div>


<div align="center">
            <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #fff;">Raining</h1>
</div>
<div style="text-align:center;">
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/webps/rgb_rain.webp?raw=true" width="400"/>
  </div>
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/webps/event_rain.webp?raw=true" width="400"/>
  </div>
</div>

<div align="center">
            <h1 style="font-size: 2rem; margin-bottom: 1rem; color: #fff;">Indoor</h1>
</div>
<div style="text-align:center;">
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/rgb_indoor.gif?raw=true" width="400"/>
  </div>
  <div style="display:inline-block; margin:0 10px;">
    <img src="https://github.com/miccunifi/FRED/blob/main/static/videos/event_indoor.gif?raw=true" width="400"/>
  </div>
</div>



---

## ✨ Citation

If you use **FRED** in your research, please cite:

```
@inproceedings{magrini2025fred,
  title={FRED: The Florence RGB-Event Drone Dataset},
  author={Magrini, Gabriele and Marini, Niccol{`o} and Becattini, Federico and Berlincioni, Lorenzo and Biondi, Niccol{`o} and Pala, Pietro and Del Bimbo, Alberto},
  booktitle={Proceedings of the 33rd ACM International conference on multimedia},
  year={2025}
}
```


## License

This project is licensed under the [Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](https://creativecommons.org/licenses/by-nc-sa/4.0/).

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)
