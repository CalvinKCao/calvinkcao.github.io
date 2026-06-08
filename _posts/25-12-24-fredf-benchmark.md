---
layout: post
title: Do Frequency-Decomposed Time Series Losses Generalize Across Architectures?
date: 2025-12-24 21:09:00
description: 
categories: quick-experiments
thumbnail: assets/img/9.jpg
---

[Technical Write-up](https://github.com/CalvinKCao/fredf-empirical/blob/main/WRITEUP.pdf) | [GitHub](https://github.com/CalvinKCao/fredf-empirical)

This was a group project for my Theory of Machine Learning course at Western University, done alongside Guo Jia, Thomson Lam, and Alex Caraman. The core question was about generalization: specifically, whether a loss function that improves one model architecture will reliably improve another. We tested the FreDF frequency-domain loss function on both iTransformer and TSMixer across four datasets, treating it as a probe for how robust the claimed benefits of frequency-domain optimization actually are. FreDF improved iTransformer consistently, but the results on TSMixer exposed a bit of brittleness. On the ILI influenza dataset, applying FreDF caused MSE to jump from 2.19 to 16.43. Spectral analysis suggests  ILI lacks the concentrated periodic structure that FreDF assumes, and forcing the model to optimize for frequency peaks that do not exist in the data causes it to overfit to noise instead. God example of a method that generalizes poorly when its inductive bias does not match the data distribution. The takeaway is that preprocessing steps like spectral energy analysis should probably gate loss function selection, rather than treating any single objective as a universal improvement.