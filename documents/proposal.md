### Title: Summarizes the main idea of your project.
Sample-Efficient play of Atari Games with Multi-Modal input

### Who: Names and logins of all your group members.
- Noah Rousell: `nrousell`
- Ryan Eng: `reng1`
- Julian Beaudry: `jbeaudry`
- Adithya Sriram: `asrira17`

### Introduction: What problem are you trying to solve and why?
We are implementing the paper [Bigger, Better, Faster: Human-level Atari with human-level efficiency](https://arxiv.org/abs/2305.19452) and adding audio inputs as well. The original model, `BBF` was able to learn many Atari 2600 games at a superhuman level with only 2-hours of game data.

We will implement a similar sample-efficient architecture that also has the capacity to intake audio as input. We plan to train our model on the Atari 2600 game *Amidar*, where audio is useful for the player.

### Related Work: Are you aware of any, or is there any prior work that you drew on to do your project?
- [2015 DQN Paper](https://www-nature-com.revproxy.brown.edu/articles/nature14236)
  - This paper introduced Deep Q-Learning, which uses neural networks to perform Q-learning directly on high-dimensional sensory inputs. This removes the need for hand-engineered features on many Reinforcement Learning tasks.
- [BBF Paper](https://arxiv.org/pdf/2111.00210.pdf)
    - This 2023 paper is the current SOTA on sample-efficient RL algorithms for Atari 2600 games. It and [EfficientZero](https://arxiv.org/abs/2111.00210) are the two architectures that have been achived human-level performance with just two hours of gameplay. Our plan is to follow mostly along with the architecture and training scheme of the BBF paper and to add audio inputs as well.
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
  - [PR that added audio support](https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/233)git
- [CrossModal Attentive Skill Learner](https://arxiv.org/pdf/1711.10314.pdf)
    - This paper develops an architecture that attends to both video and audio inputs to train a model to play the Atari games *Amidar* and *H.E.R.O.*. They also added support for audio to the [Arcade Learning Environment](https://github.com/Farama-Foundation/Arcade-Learning-Environment) (represented in the PR above), which we will take advantage of.

### Data: What data are you using (if any)?
We plan on creating a dataset similar to the Atari 100k benchmark dataset, which consists of 100 thousand observations from an Atari game with a frame-skip of 4 (around 2 hours of game-time). The only change will be that we will also incorporate audio into the model. We will use the Ar

### Methodology: What is the architecture of your model?
**Overall Architecture**
Our model architecture will be very similar to the BBF model, a deep (and wide) ResNet with residual connections. We will also apply similar regularization techniques as the ones described in the BBF paper: periodic hard weight resets at later layers, increasing the discount hyperparameter during training, and weight decay (L2 regularization). Notably, we will also include audio observations (unlike the original BBF model) and process it with convolution layers similarly to the video input. 

**Experiments**
We plan on running several experiments: training a model with only video input, training a model with both video and audio, and training a model on the RAM exposed by the game (if there is time). If possible, we also plan to perform ablation and other interpretation experiments to determine the effect of audio input.

**Backup ideas**
If computation becomes intractable, we plan to implement our architecture on a simpler Reinforcement Learning task that we will find or create ourselves (ex. a version of Pong where audio signals an important speed-up of velocity). 

**Goals**
- **Base**: implement the architecture with vidoe & audio on a simple video/audio-based game
- **Target**: implement the architecture with video & audio on Amidar
- **Stretch**: Implement the architecture with video & audio on multiple Atari games and perform interpretation
- 
### Ethics: Choose 2 of the following bullet points to discuss
**How are you planning to quantify or measure error or success? What implications does your quantification have?**
In our project, we aim to gauge the efficacy of our model by quantifying its performance in playing the game. We will select a game that offers a quantifiable metric to delineate player success. To illustrate, consider Pacman, where advancement through various levels correlates with surviving successive rounds amidst increasing ghost challenges and game complexities. In this context, we intend to measure our model's success by pinpointing the round it manages to survive. Moreover, if our model performs well enough, we could benchmark our model against human proficiency. By comparing the model's performance with that of a human player, we aim to assess its ability to emulate human-level gameplay. Additionally, by investigating whether playing with sound influences the model's learning process, we delve into the role of auditory cues in gaming. Should the model without sound outperform its counterpart with sound, it would suggest that auditory elements might not significantly impact gameplay progression in the selected game. Such insights could inform game design considerations and shed light on the interplay between sensory modalities in gaming experiences.

**Why is Deep Learning a good approach to this problem?**
Deep learning is a good approach to this problem because it is a very high dimensional problem with many weights and inputs that need to be considered. Additionally, deep learning allows for feature learning where the model can learn useful features from raw sensory input, such as pixels in the game frames, without manual feature engineering. Deep learning models can learn hierarchical representations of data. In the context of games, this means that the model can learn to represent complex game states in a way that captures both low-level details like pixel values and high-level concepts such as game dynamics. 


### Division of labor: Briefly outline who will be responsible for which part(s) of the project.
- All: impementing the architecture
- Misc. tasks
  - Noah: research & interpretation
  - Adithya: creating and processing the dataset
  - Ryan & Julian: creating a *simple* test video/audio game that adheres to the Gym interface
