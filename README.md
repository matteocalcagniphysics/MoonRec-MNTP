# Computational-Physics-2026

This repository contains the work carried out for a lunar segmentation project aimed at identifying and analyzing geological features on the Moon from image data. The project combines computer vision, deep learning, and visualization to tackle two related tasks:

- semantic segmentation, to classify each pixel of an image into a geologic class;
- instance segmentation and panoptic understanding, to detect and separate individual objects such as craters or other distinct features.

The overall goal is to build a pipeline that can take lunar imagery, preprocess it into usable training tiles, train segmentation models, generate predictions, and visualize the results in a way that is useful for scientific analysis.

## Team contributions

| Person | Main contribution |
| --- | --- |
| Matteo Renato Calcagni | Developed Panoptic Network |
| Pasquale Andreacchio | Developed Mask R-CNN |
| Agostina Gonzati | Preprocessing pipeline and U-Net |
| Nicola Lavarda | Visualization, result inspection, and qualitative analysis |

## Project overview

The repository is organized around the idea of building a complete lunar segmentation workflow. The pipeline starts from raw lunar imagery and ends with predictions that can be inspected visually and evaluated quantitatively.

The work includes:

1. collecting and preparing image tiles and labels;
2. transforming raw data into training-ready samples;
3. developing semantic segmentation models for pixel-wise classification;
4. developing instance segmentation models for object-level detection;
5. integrating panoptic-style prediction strategies;
6. generating visual overlays and analysis tools for interpretation.

The project is particularly relevant for planetary science because lunar surface features are often small, irregular, and embedded in complex textures. This makes them challenging for standard segmentation approaches and motivates the use of modern deep learning methods.

## Repository structure

The repository contains the following main areas:

- data: utilities for preparing and loading dataset samples;
- models: implementations of segmentation and panoptic architectures;
- training: scripts and trainers for model optimization;
- inference: prediction pipelines for applying trained models;
- visualization: tools for generating plots and overlays;
- notebooks: exploratory experiments and demonstrations.

The implementation is organized around modular components so that each part of the pipeline can be developed and tested independently.

## Typical workflow

A typical run of the project follows this path:

1. Load lunar imagery and labels.
2. Preprocess them into tiles and masks.
3. Train a semantic segmentation model such as U-Net.
4. Train or configure an instance segmentation approach such as Mask R-CNN.
5. Combine semantic and instance outputs in the Panoptic Network.
6. Generate predictions on new tiles.
7. Visualize the outputs and assess quality.

This workflow makes the project suitable for both experimentation and reproducible development.

## Expected outputs

The repository is intended to produce several kinds of outputs:

- semantic segmentation masks;
- object-level masks from Mask R-CNN;
- panoptic-style predictions combining both perspectives;
- visual comparisons between input images and model outputs;
- analysis tools to inspect results qualitatively.

These outputs are useful both for model development and for communicating the results of the project.

