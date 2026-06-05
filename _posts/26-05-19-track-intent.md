---
layout: post
title: Do RL Agents Track Each Other's Intent? (Part 1)
date: 2026-05-19 20:01:02
description: Linear probes decode partner intent from RL hidden states, but architecture determines whether any of it is interpretable.
tags: formatting images
categories: quick-experiments
thumbnail: assets/img/9.jpg
---

> **TLDR:** Building on a recent paper, I wanted to see if reinforcement learning agents build internal models of their partners during cooperative tasks. I built a 9x7 gridworld where a linear probe successfully decoded one agent tracking another's hidden intent. When I tried scaling this pipeline to more complicated tasks like Overcooked, architectural mismatch broke the interpretability. Environment incentives and recurrent state shapes dictate whether a multi-agent system is actually interpretable and will inform the next steps of this project.

## Agent coordination

Humans are naturally great at reading each other. You watch a coworker walk toward the printer and instantly infer what they're trying to do. Reinforcement learning agents coordinate in shared environments, but we don't have a clear mechanistic understanding of how. Policies are usually black boxes mapping observations to actions.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/bush-planning.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Bush et. al (2025) find mechanistic evidence of agents planning movement (top) and box pushes (bottom) in a Sokoban environment. Adapted from original paper.
</div>

Bush et al. (2025) recently provided the first mechanistic evidence that model-free RL agents learn to plan internally. They studied a Deep Repeated ConvLSTM agent playing solo puzzle game Sokoban. Because a ConvLSTM maintains a 3D structure that allows for easy matching of features game board squares, the authors could apply linear probes to specific cell states and read out exactly where the agent planned to move and push boxes.

I wanted to apply this idea to multi-agent settings. If Agent A coordinates with Agent B, does Agent A's hidden state encode a decodable, belief-like variable about Agent B's intent?

## Quick proof of concept

I built a simple 9x7 grid-based world. Two agents spawn in a shared bottom zone and must navigate to separate goals located at the end of three possible vertical corridors, left/centre/right; L=0, C=1, and R=2.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/3-corridor.svg" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Layout of the three-corridor environment on a 9x7 grid, featuring spawn zone at bottom, three bottleneck corridors in the middle, and three goal zones across the top (L, C, R). The green trajectories illustrate a successful trial where agents coordinate to reach separate goals without bumping together. The red trajectories represent a failure case where both agents both try going through the centre bottleneck.
</div>

At the start of every episode, each receives a private goal assignment. Agent A receives its own goal as a 3D one-hot vector, but it never observes Agent B's. Agent A has to infer it by watching Agent B move. If they enter a bottleneck simultaneously, they collide and get penalized.

I trained the agents using the CTDE-lite variant of Multi-Agent PPO with parameter sharing between both agents. Since the usual feedforward policies can't track partial observables over time, I used a modified version of the DRC(3,3) architecture. The convolutional encoder preserves the spatial dimensions with padding. The state $h$ stays a $32 \times 7 \times 9$ tensor. I also used a pool-and-inject mechanism that globally pools $h$, passes it through a linear layer, and adds it back before the gate computation.

Standard PPO implementations usually truncate gradients at episode boundaries. If you want a ConvLSTM to accumulate evidence about a partner, you have to use Backpropagation Through Time across the full rollout. So, I ran 20-step rollouts and calculated advantages using Generalized Advantage Estimation.

## Probing hidden states

Bush et al. probed their agent at the level of individual action plans. At each grid cell, they asked whether the agent planned to step onto the square or push a box from it. The granularity works in Sokoban because the objective is concrete and spatially decomposable into single-step box displacements.

My planned environments for the next phase, including Overcooked, have messier spatial semantics. The reward structure is more abstract: multi-step food preparation chains replace the push-box-onto-goal signal, and individual tiles carry less interpretable meaning. So rather than probing at the level of single-step moves, I wanted to probe for something higher-level: which corridor would Agent B choose to navigate toward? This is closer to a belief or intent variable than an action plan, and felt like a more appropriate target for environments where objectives are compositional. This turned out to be overly optimistic, for reasons I will get to.

I collected greedy rollouts and trained 1x1 logistic regression probes on the layer-D cell state at specific timesteps. I used L-BFGS with 2000 iterations to predict the partner's target corridor. At step zero, the macro F1 score was $0.344$, matching random chance. By step five it climbed to $0.903$. The agent was tracking its partner's intent as the episode progressed.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/confusion_matrix.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Matrix of accuracy of linear probe trained to predict the partner agent's intended corridor (Left, Center, Right) from the model's internal activations at step 4. Strong diagonal values and overall 75.5% accuracy demonstrate that the agent successfully learns to infer/represent its partner's goals.
</div>

To test whether the agent actively relied on this representation, I attempted a causal intervention. I extracted the weight vector $W_k$ from the trained probe, scaled it by a factor of $\alpha=1.0$ to match the norm of the cell state, and injected it back into Agent A's network during a live episode.

The intervention failed to alter the long-horizon policy. The immediate action distribution shifted slightly; I measured a positive KL divergence at the exact step of injection, but the agent ignored the injected belief. 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/intervention_kl.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    Policy shift (KL divergence) following hidden state intervention. Only injections targeting the RIGHT corridor produced a behavioral shift stronger than a random control vector (1.35×).
</div>

The most likely explanation is geometric. Logistic regression finds the direction most predictive of a concept in a linear sense, but this is not necessarily the direction the policy reads. The cell state feeds into subsequent gate activations nonlinearly before hitting a final 2016 to 256 linear projection. The policy's active decision subspace occupies a different region of the high-dimensional tensor than the one my probe identified. The policy likely encoded partner intent diffusely across many weakly correlated dimensions rather than in a single clean direction. A corridor-choice variable is more distal from any specific gate computation than an action-level variable, so the orthogonality problem is worse for high-level goal probes like mine than for the action-level probes in Bush.

Even so, I decided to treat the F1 result as a positive existence proof and move on to scaling. An F1 of 0.903 by step five establishes the hidden state genuinely contains recoverable information about partner intent. Refining the intervention method on a nine-by-seven grid felt like diminishing returns when the more pressing question was whether this interpretability pipeline would survive a more complex environment at all.

## Scaling up, breaking down

I wanted to replicate this in a more complex setting. My first thought was multi-agent Boxoban, but quickly found that when you train two agents on standard Sokoban levels designed for single players, the joint policy collapses. One agent solves the puzzle while the second agent stands entirely still in a corner to avoid incurring step penalties. As far as I could tell, no 2-player Sokoban level datasets existed.

To bypass the compute bottleneck of training a custom curriculum, I pivoted to the Overcooked task (Carroll et al.). 

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/overcooked.png" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>
<div class="caption">
    In Overcooked, the objective is to sell as much soup as possible. This consists of delivering three onions to a pot, then bringing soup on a dish to customers. Adapted from Shah et al. (2019). 
</div>

I used pre-trained PPO checkpoints from the ZSC-Eval model zoo on the random0 layout. Quickly, I hit a wall, realizing the Overcooked models used a standard GRU instead of a ConvLSTM. The convolutional encoder took the 5x5 board observation and crushed it down into a flat 64-dimensional feature vector before it ever reached the recurrent layer.

<div class="row mt-3">
    <div class="col-sm mt-3 mt-md-0">
        {% include figure.liquid loading="eager" path="assets/posts/26-05-19-track-intent/architecture_comparison.svg" class="img-fluid rounded z-depth-1" zoomable=false %}
    </div>
</div>

This flat representation also ruined any chance of clean causal interventions. In the single-agent Sokoban paper, the authors probed specific spatial coordinates. In my Overcooked setup, the 64-dimensional vector represents the entire state. I extracted the weights corresponding to the dish concept and injected it into the ego agent. The team score dropped by 20 points, but in terms of understanding why at a more granular level, I was stuck. Without spatial grounding, it is impossible to isolate what the agent perceived. The intervention forces the agent into a generic dish mode without specifying where the dish is located or who is holding it.

## The path forward

I think moving forward I have two options. 1) I could return to the ConvLSTM architecture. Mechanistic interpretability in spatial environments given current techniques requires the hidden state to map directly to grid cells. I'll need to write a procedural generation algorithm to build Sokoban levels that strictly require two players to push separate boxes simultaneously. 2) I could try to use sparse autoencoders to see if I can decompose the flat 64-element GRU representation into cell-specific planning features.

Another thing I realized is that even if the ConvLSTM architecture or SAEs work, l need to train the agents without parameter sharing. Using MAPPO with shared weights introduces a massive confound for theory-of-mind research. If both agents share the same neural network, any decodable intent is potentially an artifact of the shared parameters rather than genuine observation-based inference. Setting up RL environments for interpretability research demands rigorous engineering, and I have some rebuilding to do.

## References

Bush, T., Chung, S., Anwar, U., Garriga-Alonso, A., & Krueger, D. (2025). Interpreting Emergent Planning in Model-free Reinforcement Learning.

Carroll, M., Shah, R., Ho, M. K., Grifﬁths, T. L., Seshia, S. A., Abbeel, P., & Dragan, A. (2019). On the Utility of Learning about Humans for Human-AI Coordination.
