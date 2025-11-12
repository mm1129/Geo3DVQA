<h2 align="center">
  Geo3DVQA: Evaluating Vision-Language Models for 3D Geospatial Reasoning
from Aerial Imagery
</h2>


Three-dimensional geospatial analysis underpins critical applications in urban planning, climate adaptation, and environmental assessment. Current methods rely on expensive specialized sensors (e.g., LiDAR and multispectral) that limit global accessibility. Existing sensor-based and rule-driven methods further struggle with tasks that require the integration of multiple 3D cues, handling uncertainty, and providing interpretable reasoning. We introduce **Geo3DVQA**, the first comprehensive benchmark for evaluating vision–language models (VLMs) in height-aware 3D geospatial reasoning from RGB-only remote sensing imagery. Unlike traditional sensor-based frameworks, Geo3DVQA emphasizes holistic scenarios that combine elevation, sky view factors, and land cover patterns. The benchmark includes 110k curated question–answer pairs spanning 16 task categories across three complexity levels: single-feature inference, multi-feature reasoning, and application-level spatial analysis. The evaluation of ten state-of-the-art VLMs highlights the difficulty of RGB-to-3D reasoning. GPT-4o and Gemini-2.5-Flash achieved only 28.6\% and 33.0\% accuracy respectively, while domain-specific fine-tuning of Qwen2.5-VL-7B achieved 49.6\% (+24.8 points). These results reveal the current VLM limitations and establish a new challenge frontier for scalable and accessible 3D geospatial analyses with holistic reasoning capabilities.

![dataset](dataset.png)

## News
- 2025/11/28, Preparing the dataset/code release and camera-ready materials

## ToDo
- [ ] update code & data

## Data preparation

I'll provide the dataset link soon.
The small example containing 3000 QAs (short answer + freeform) are in ![inference folder](inference/svf_combined_3000q_free_example.jsonl).

## Evaluation Protocol
- Decoding: temperature=0 (greedy) unless otherwise noted. When `temperature>0`, report mean±std over 3 runs (appendix).
- Multiple-choice scoring: category-specific rules implemented in `inference/calc_acc_res.py` (e.g., Jaccard≥0.7 for `landcover_type`, order-independent exact set for `land_use`, 30% relative or <10 absolute tolerance for `height_average`, ±0.05 for `hard_pixel`).
- Free-form evaluation: GPT-based automatic scoring with temperature=0.0; NaN assigned when the corresponding GT domain is absent. Evaluation traces are saved.

## Disclaimer
**Important**: This repository contains a refactored reference implementation of the method described in our paper. For readability and maintainability, the code has been reorganized and some parts might be modified or added for additional experiments after the paper was written.
However, the implementation reflects the same core methodology and is intended as a basis for reproducing and extending our work.

## Citation
```text
% Add citation here (BibTeX/Reference)
```

## Acknowledgments
This project builds upon public geospatial datasets and community tools in remote sensing and multimodal learning. We thank the maintainers and contributors of the referenced projects.

We used GeoNRW dataset for RGB, DSM (digital surface model), segmentation.
GeoNRW is available on ![SynRS3D](https://github.com/JTRNEO/SynRS3D).
SVF (sky view factor) was generated using UMEP from the DSM.

## License
All images and their associated annotations in Geo3DVQA can be used for academic purposes only, but any commercial use is prohibited.

