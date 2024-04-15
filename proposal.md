### Title: Summarizes the main idea of your project.
Sample-Efficient play of Atari Games with Multi-Modal input

### Who: Names and logins of all your group members.
- Noah Rousell: `nrousell`
- Ryan Eng: `reng1`
- Julian Beaudry: `jbeaudry`
- Adithya Sriram: `asrira17`

### Introduction: What problem are you trying to solve and why?
We are implementing the paper [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452) and adding audio inputs as well. The original model, `BBF` was able to learn many Atari 2600 games at a superhuman level with only 2-hours of game-time.

We will implement a similar sample-efficient architecture that also has the capacity to intake audio as input. We plan to train our model on the Atari 2600 game *Amidar*, where audio is useful for the player.

### Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?
- [2015 DQN Paper](https://www-nature-com.revproxy.brown.edu/articles/nature14236)
  - This paper introduced Deep Q-Learning, which uses neural networks to perform Q-learning directly on high-dimensional sensory inputs. This removes the need for hand-engineered features on many Reinforcement Learning tasks.
- [BBF Paper](https://arxiv.org/pdf/2111.00210.pdf)
    - This 2023 paper is the current SOTA on sample-efficient RL algorithms for Atari 2600 games. It and [EfficientZero](https://arxiv.org/abs/2111.00210) are the two architectures that have been achived human-level performance with just two hours of gameplay.
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
  - [PR that added audio support](https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/233)
- [CrossModal Attentive Skill Learner](https://arxiv.org/pdf/1711.10314.pdf)
    - This paper develops an architecture that attends to both video and audio inputs to train a model to play the Atari games *Amidar* and *H.E.R.O.*. They also added support for audio to the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) (represented in the PR above), which we will take advantage of.

### Data: What data are you using (if any)?
We plan on using the Atari 100k benchmark dataset, which consists of 100 thousand observations from an Atari game with a frame-skip of 4 (around 2 hours of game-time). 

### Methodology: What is the architecture of your model?
- mostly the BBF model: ResNet model like Impala-CNN but 4x as wide
  - Different regularization and annealing applyied during training: weight decay, hard resets every 40k training steps, increase of gamma (discounting hyperparameter) during training, 
  - Audio as well, probably with convolution layers too at the beginning
- Experiments
  - only video (normal BBF)
  - video + audio
  - RAM?
  - ablation / interpretation
- Backup ideas
  - Finding/constructing simpler RL tasks if Atari 2600 games prove to be too much computation
- Goals
  - Base: implement the architecture with vidoe & audio on a simple video/audio-based game
  - Target: implement the architecture with video & audio on Amidar
  - Stretch: Implement the architecture with video & audio on multiple Atari games and perform interpretation

### Ethics: Choose 2 of the following bullet points to discuss
- **What broader societal issues are relevant to your chosen problem space?**
- **Why is Deep Learning a good approach to this problem?**
- What is your dataset? Are there any concerns about how it was collected, or labeled? Is it representative? What kind of underlying historical or societal biases might it contain?
- Who are the major “stakeholders” in this problem, and what are the consequences of mistakes made by your algorithm?
- How are you planning to quantify or measure error or success? What implications does your quantification have?

- This project has implications in robotics: if robotics get a lot better, there is the possibility of large automation and replacement of jobs
- f


### Division of labor: Briefly outline who will be responsible for which part(s) of the project.
- All: impementing the architecture
- Misc. tasks
  - Noah: research & interpretation
  - Adithya: creating and processing the dataset
  - Ryan & Julian: creating a *simple* test video/audio game that adheres to the Gym interface
