---
layout: post
title: Four Failed Experiments in LoRA Attribution
date: 2026-06-05 11:22:34
description: Post-hoc methods fail to untangle multi-LoRA generations.
tags: experiments attribution diffusion
categories: quick-experiments
---

> TLDR: Modular image generation platforms allow users to stack multiple Low-Rank Adaptations, creating a need to attribute specific outputs to specific adapters at scale. Inspired by Mlodozeniec et al. (2024), I tested whether cheap post-hoc methods or linear approximations bypass the massive computational cost of combinatorial ablation. I ran experiments across five different methodologies, starting from simple caching tricks up to complex DDIM pseudo-ablation. They all failed to provide correct localized attributions. The non-linearities of the forward pass and the trajectory commitment in latent space mean true single-removal ablation remains the only accurate method.
---

## Multi-LoRA attribution

On many popular AI image generation platforms, users stack multiple adapters simultaneously to customize base models, blending concepts, styles, and characters. The most popular adaptation method is Low Rank Adaptation, which simply adds the product of two low-rank matrices to a network's weight matrices, allowing for efficient fine-tuning:

$$W_{\text{final}} = W_{\text{base}} + \sum_{i=1}^{N} \alpha_i \Delta W_i$$

When an output image contains harmful or copyrighted content, or looks terribly distorted, we must identify the responsible component. Image generation platforms want to attribute policy violations to individual uploaded files, and consumers want to know which adapter is mangling the output. The impact of individual adapters in a multi-adapter stack is difficult to analyze. The forward pass is nonlinear because of cross-attention softmax, layer normalization, and GELU activations. Simple weight inspection or linear activation assumptions fall short. Currently, the main method of determining which adapter in a stack is harmful is running combinatorial ablation, which requires generating images with every possible subset of adapters to isolate their effects. Even single-removal ablation requires full generation passes, scaling poorly for real-time applications.

Mlodozeniec et al. (2024) used influence functions for scalable data attribution in base diffusion models. I wanted to find a similarly scalable attribution mechanism for multi-adapter setups, starting with the simplest possible approach and working my way up to more complex interventions as each hypothesis fell apart.

## Experiment 1 Early-layer activation caching

My first thought was straightforward. Since full ablation for every adapter takes tonnes of compute, I wanted to skip redundant calculations. Some prior literature reports that weight perturbations only affect the deeper layers of the UNet. If this was true, running the base model once and caching the early layer activations would allow me to reuse them across different adapter combo ablation passes.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-04-02-lora-attr/diagram_unet.svg" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    My initial caching setup. If adapter perturbations were truly confined to deeper layers, calculating the early blocks once and reusing those activations would let me skip redundant forward passes.
</div>

I set up a test using Stable Diffusion v1.5, running 10 images under four different configurations. I used a baseline with no adapters, a Sketch style adapter, a Pokemon character adapter, and a combination of all three at a scale of 0.8. I measured the relative activation variance across these configurations for every single UNet block. The first block showed 60.9% relative variance across the configurations, and deeper layers hit 94.1% variance. The perturbations from the adapters propagate instantly through the network. Early-layer caching is a dead end.

## Experiment 2 Sparse autoencoders

If early layers could not be cached, I wanted to see if I could decompose the activation-space perturbations directly. (I eventually realized this was naive, which I will get to in a bit.) Before running the decomposition, I set up three distinct components for testing: a global style modifier using a pre-trained freehand sketch checkpoint, a spatially localized subject using a Pokémon adapter trained on 833 captioned images, and a spatially localized object using a Backpack adapter trained on 20 images from a DreamBooth dataset.

To build a baseline of real model activations, I took 100 standard text prompts and ran the base Stable Diffusion v1.5 model for 50 denoising steps. At 25 specific timesteps during each run, I extracted the output of the first mid-block attention layer. This layer outputs a tensor with an 8x8 spatial grid, which I average-pooled to collapse the spatial layout, resulting in a single 1280-dimensional vector. This process yielded 2,500 activation vectors.

I trained a sparse autoencoder with 512 hidden dimensions and an L1 penalty of 0.001 on these average-pooled base activations to learn a dictionary of sparse latent features.

To perform the attribution, I captured activations across three conditions. I extracted a base pass, single-adapter reference passes, and a mixed pass with all adapters active. Encoding these through the autoencoder yielded sparse latent vectors. I subtracted the baselines to isolate the latent perturbations. A Ridge regression confirmed the autoencoder preserved the adapter features, mapping latent perturbations back to activation perturbations with high accuracy.

I applied an Ordinary Least Squares linear solver to decompose the mixed latent perturbation as a combination of reference perturbations. I needed to extract mixed activations from a real generation. I started with random Gaussian noise from a fixed seed and ran the denoising process using a prompt combining all three concepts, like "a pokemon creature on the left side of the image, a backpack on the right side, freehand sketch style". Halfway through the 30-step denoising process at step 15, I paused the forward pass and extracted the activation tensor flowing through the attention block. I average-pooled this tensor across the spatial grid to collapse it into a single 1280-dimensional vector. Passing this through the trained autoencoder yielded the 512-dimensional mixed latent vector. I repeated this extraction for the isolated reference components.

I set up the regression using the 512 hidden dimensions as independent samples. This created a system of 512 equations to solve for the three unknown adapter scales.

$$\Delta z_{\text{mixed}} \approx \sum \beta_i \Delta z_{\text{ref},i}$$

Using a closed-form solver, the regression failed completely. In a three-adapter mixture with true scales of 0.80, 0.40, and 0.20, the solver recovered scales of 0.73, 0.14, and 0.05. The dominant sketch style component showed a 9% error, while the weaker localized components suffered massive 65% and 75% errors. Because of the model's internal nonlinearities, the mixed activations are not a clean linear sum of the individual references. The solver forced the coefficients to align heavily with the dominant sketch style to minimize the squared error over the 512 dimensions, drowning out the signal of the weaker components.

Looking back, I should have seen this failure coming. My fundamental mistake was assuming that stacking adapters creates a clean linear sum of their individual activation features. The UNet is built on non-linear operations, particularly the softmax function inside the cross-attention blocks. Because softmax forces the outputs to sum to one, a dominant style adapter that pushes its weights higher will inherently squash the values of any weaker components in that same layer. The mixed activation state is never a simple superposition of the isolated references. Expecting an ordinary linear equation solver to untangle a non-linear normalization process was doomed to fail.

## Experiment 3 Per-LoRA-delta gradient attribution

If the forward pass is a nonlinear mess, looking backward at the gradients seemed like a viable alternative. I adapted discriminative gradient attribution to weight-delta space. I defined a spatial bounding box around a specific subject, computed the MSE loss of the noise prediction inside the box, and backpropagated the loss to the adapter weights. To remove weight-magnitude bias, I divided the scores by the Frobenius norm of the weights.

I tested this approach on 30 images generated with the Sketch, Pokemon, and Backpack adapters. I used custom spatial layout prompts designed to force physical separation, such as "a pokemon creature on the left side of the image, a backpack on the right side". I manually annotated 99 bounding boxes to separate the subjects.

The top-1 accuracy for the Pokemon bounding box was exactly 0%. The mean attribution shares inside the boxes were almost perfectly uniform across all three adapters, hovering around 33.3% each. My best guess for why this happens is that the spatial signal washes out completely during backpropagation. The attention projection weights act globally on latent states. When you spatially mask the loss at the output and backpropagate, the localized signal disperses globally across the projection matrices.

## Experiment 4 Invert-once pseudo-ablation

This led to my final attempt. If true single-removal ablation is too expensive because it requires full generation passes from pure noise, simulating ablation directly in latent space partway through the process felt promising.

To evaluate this, we need to distinguish between true ablation and pseudo-ablation. A true ablation map is generated by running a complete image generation from pure noise with one adapter entirely removed, then comparing the final output to the full-stack generation to isolate the adapter's impact. A pseudo-ablation map attempts to shortcut this. I inverted the generated image via DDIM inversion to cache intermediate latents at various timesteps. To approximate the removal of a specific adapter, I set its scale to zero and ran a short 10-step denoising path starting from that cached latent. I calculated an attribution ranking by determining which adapter's removal caused the largest deviation from the original image.

I compared these pseudo-removal maps to true ablation maps across 20 general-purpose prompts, using prompts like "a cat sitting on a wooden table" to run the baseline generations. Unfortunately, the spatial correlation between the pseudo and true maps was non-existent. The top-1 attribution ranking accuracy hit 3.75%, performing worse than a random guess. I ran a diagnostic sweep increasing the denoising steps up to 30, and the correlation decreased monotonically.

I suppose the root cause here is trajectory commitment. A cached latent at step 14 is not a neutral state. It is structurally committed to the joint composition of all active adapters. Resuming denoising with one adapter missing forces the model to complete a trajectory starting from an inconsistent layout. The cached latent fails to converge to the true ablated trajectory.

## Future work

Post-hoc linear methods and cached latent denoising are mathematical dead-ends for this specific problem. Sparse autoencoders to quantify  localized contribution in a mixed setup. True single-removal ablation remains the only accurate method for multi-adapter environments. It is possible that multi-adapter setups are fundamentally non-untangleable. Mechanistic interpretability circuit analysis to map specific pathways directly inside the UNet remains a stretch goal for future work.

## Sources

Mlodozeniec, B., Eschenhagen, R., Bae, J., Immer, A., Krueger, D., & Turner, R. (2024). Influence Functions for Scalable Data Attribution in Diffusion Models. [https://doi.org/10.32388/BOJDXM](https://doi.org/10.32388/BOJDXM)