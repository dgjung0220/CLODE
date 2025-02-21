### Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs <br> (ICLR 2025 Spotlight)

This repository is the official implementation of "Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs" @ ICLR25.

Donggoo Jung*, [Daehyun Kim](https://github.com/kdhRick2222)*, [Tae Hyun Kim](https://scholar.google.co.kr/citations?user=8soccsoAAAAJ) $^\dagger$  (\*Equal Contribution, $^\dagger$ Corresponding author)

[[ICLR2025] Paper](https://openreview.net/forum?id=Mn2qgIcIPS)

## Method
![Main_Fig.](assets/main_figure.png)
We propose the unsupervised low-light image enhancement problem by reframing discrete iterative curve-adjustment methods into a continuous space using Neural Ordinary Differential Equations (NODE).

| Under-exposure | Over-exposure | Normal-exposure | 
| :------------: | :-----------: | :-------------: |
| <video src="https://github.com/dgjung0220/CLODE/assets/-" /> | <video src="https://github.com/dgjung0220/CLODE/assets/-" /> | <video src="https://github.com/dgjung0220/CLODE/assets/-" /> |

## Evaluation

Download the pre-trained [model](https://drive.google.com/drive/folders/1E1Oi89TJeZIL4pz7d4p-D_Yq1rAG4Uhc?usp=drive_link) and place it in ``./pth/``

```bash
# In inference.py, only modify the following paths:
# file_path: Path to the input images
# gt_path: Path to the ground truth images
# file_path = '/path/to/your/input'
# gt_path = '/path/to/your/corresponding_gt'

$ python inference.py
```

## User Controllablity
CLODE learns the low-light exposure adjustment mechanism in the continuous-space, and is trained to output $I_T$ by integrating the states from $0$ to $T$ using a fixed $T=3$. However, users can manually adjust the integration interval by changing the final state value $T$ at the test stage, allowing them to output images with the preferred exposure level and even produce images darker than the input. In practice, by controlling the final state from $-(T+\Delta t)$ to $(T+\Delta t)$, the exposure level of the output image can be easily controlled to provide a more user-friendly exposure level. 

```bash
$ python inference.py --T 4.8    # set to 3.5, more brighten
$ python inference.py --T -1.4   # set to -1.4, more darken
$ python inference.py --T 2.5    # set to 2.5, Adjust to the brightness desired by the user
```

## Results
![Main_Fig.](assets/result_figure.png)

We provide our results for the LOL and SICE Part2 dataset. (CLODE/**CLODE**$\dagger$)
| Dataset | PSNR | SSIM | Images|
| :------:| :---:| :---:| :---: |
| LOL | 19.61/**23.58** | 0.718/**0.754** | [Link](https://drive.google.com/drive/folders/1Xsalp32GyNEG6tabPVKrRZ2WD69m32oP)/[Link](https://drive.google.com/drive/folders/14r8x7C6ERXjCDtug63MakbJ3IXgiua6o) |
| SICE | 15.01/**16.18** | 0.687/**0.707** | [Link](https://drive.google.com/drive/folders/1uf9WDFhmCRiVwFDR8X5fj4Zm5byAQ-Yw)/[Link](https://drive.google.com/drive/folders/1hysjI2Xt9NYqvDgV2DHMAlK7RGj6HO0g) |

## Train

```
python main_experiment.py
```

## Citation
If you find our work useful in your research, please consider citing our paper.
```bibtex
@article{jung2025continuous,
  title={Continuous Exposure Learning for Low-light Image Enhancement using Neural {ODE}s},
  author={Donggoo Jung and Daehyun Kim and Tae Hyun Kim},
  booktitle={ICLR},
  year={2025},
}
```

## Acknowledgement
We are using [torchdiffeq](https://github.com/rtqichen/torchdiffeq) as the Neural ODEs library. We thank the author for sharing their codes.