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
