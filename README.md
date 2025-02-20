### Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs <br> (ICLR 2025 Spotlight)

This repository is the official implementation of "Continuous Exposure Learning for Low-light Image Enhancement using Neural ODEs" @ ICLR25.

Donggoo Jung*, [Daehyun Kim*](https://github.com/kdhRick2222), [Tae Hyun Kim](https://scholar.google.co.kr/citations?user=8soccsoAAAAJ) (\*Equal Contribution)

[[ICLR2025] Paper](https://openreview.net/forum?id=Mn2qgIcIPS)

## Method
![Main_Fig.](assets/main_figure.png)
We propose the unsupervised low-light image enhancement problem by reframing discrete iterative curve-adjustment methods into a continuous space using Neural Ordinary Differential Equations (NODE).

| Under-exposure | Over-exposure | Normal-exposure | 
| :------------: | :-----------: | :-------------: |
| <video src="https://github.com/dgjung0220/CLODE/assets/-" /> | <video src="https://github.com/dgjung0220/CLODE/assets/-" /> | <video src="https://github.com/dgjung0220/CLODE/assets/-" /> |

## TODO:

- Create notebook for Controllability
  

## Evaluation

```bash

```

## User Controllablity

```bash

```

## Results
![Main_Fig.](assets/result_figure.png)

We provide our results for the LOL and SICE Part2 dataset. (CLODE/**CLODE**$\dagger$)
| Dataset | PSNR | SSIM | Images|
| :------:| :---:| :---:| :---: |
| LOL | 19.61/**23.58** | 0.718/**0.754** | [Link]() |
| SICE | 15.01/**16.18** | 0.687/**0.707** | [Link]() |

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
