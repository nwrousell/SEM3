## Introduction
We are implementing the paper [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452) and adding audio inputs as well. The original model, `BBF` was able to learn many Atari 2600 games at a superhuman level with only 2-hours of game data.

We are implementing a similar sample-efficient architecture that also has the capacity to intake audio as input. We plan to train our model on the Atari 2600 game *Amidar*, where audio is useful for the player.

## Challenges: What has been the hardest part of the project you’ve encountered so far?
The hardest parts of the project so far have been project organization and re-implementing parts of the BBF codebase. There are certain parts of the BBF model that were not specified clearly in the paper; reading through the codebase thoroughly has been very useful. However, the codebase is somewhat overly-complex for our use case and doesn't use TensorFlow, so we've also had to learn how to simplify and adjust. As for project organization, we've found that it's much easier to reason and work on our project after refactoring and making things modular!

## Insights
We haven't ran the model for a full-run as of now, so we can't yet speak about full results (stay tuned!). 

## Plan: Are you on track with your project?
We are on track! We're ready to run the base model now (and have the environment set up on Oscar CCV). Next, we're going to add audio support and test out various architectures to deal with the audio, which we don't foresee as being as complex as the base model / training setup which is already done.


## What do you need to dedicate more time to?
Going forward, we need to spend more time reading related papers and thinking about possible ways to approach, improve, and extend different parts of the model (notably, the audio and attention bits).

## What are you thinking of changing, if anything?
We were originally thinking of modelling the audio portion of our architecture after a paper ([CrossModal Attentive Skill Learner](https://arxiv.org/pdf/1711.10314.pdf)) that already does a similar thing (albeit not in a data-efficient way), but we may diverge more now. After reading the paper and looking through their codebase more thoroughly, and thinking about how we can integrate the ideas, we have some different ideas for how to implement the attention mechanism (or how to integrate audio withou an attention mechanism at all). 