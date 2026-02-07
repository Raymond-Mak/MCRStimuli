## MCRStimuli Framework
MCRStimuli is a two-stage framework for visual emotion analysis that **self-generates multi-level sentiment stimuli** from an image and then **fuses them for robust emotion prediction**.

In Stage 1, a vision-language model is prompted to decode
**pixel-level** (global color/lighting),
**semantic-level** (human/scene/object affective cues), and
**cognitive-level** (emotion-oriented reasoning conditioned on previous stimuli) descriptions.

In Stage 2, we encode the image with a CLIP ViT image encoder and the stimuli with a RoBERTa text encoder, and use a lightweight Transformer to dynamically combine complementary evidence—where the decoded stimuli also serve as interpretable rationales for the final prediction.
![MCRStimuli Framework](framework_01.jpg)

## Generated Stimuli
The stimuli generated in Stage 1 of the MCRStimuli Framework have been placed in Google Drive [download_stimuli](https://drive.google.com/drive/folders/1pJDQA1yTFtv3nzqq2INymdPNz8m1KsDh?usp=drive_link)

### Dataset
* **EmoSet**: the largest benchmark with **118,102** images, a balanced **8-class** label space based on the Mikels model, and **10** human annotators per image. 
* **FI**: collected from Flickr/Instagram by searching Mikels emotion keywords, with **22,683** images and **8** categories that can be grouped into positive vs. negative. 
* **EmotionROI**: retrieved from Flickr using six Ekman emotion words (and synonyms), **class-balanced** at **330** images per emotion for **1,980** images total, and also supports a positive/negative split. 
* **Twitter I**: a small Twitter-sourced benchmark labeled as positive/negative, containing **1,269** images with **5** AMT annotators per image. 
* **Twitter II**: another Twitter benchmark labeled as positive/negative, containing **603** images with **3** AMT annotators per image. 
