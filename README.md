# DL-final-project

## Resources
- [2015 DQN Paper](https://www-nature-com.revproxy.brown.edu/articles/nature14236)
- [EfficientZero Paper](https://arxiv.org/pdf/2111.00210.pdf)
  - Built on top of [MuZero](https://arxiv.org/pdf/1911.08265.pdf)
  - Slightly beaten in 2023 by [BBF](https://arxiv.org/pdf/2305.19452v3.pdf)
- [OpenAI Gymnasium](https://gymnasium.farama.org/)
  - [PR that added audio support](https://github.com/Farama-Foundation/Arcade-Learning-Environment/pull/233)
  - [CASL fork of ALE with audio support](https://github.com/shayegano/Arcade-Learning-Environment)
- [CrossModal Attentive Skill Learner](https://arxiv.org/pdf/1711.10314.pdf)
    - Amidar might be a good game to use

## Environment setup
- `python -m venv env`
- `source env/bin/activate`
- `pip install tensorflow python-speech-features scikit-image imageio matplotlib pyyaml opencv-python`


- AutoROM to download ROMs manually (if using ALE instead): [docs](https://pypi.org/project/AutoROM/)

- `tf-agents` not playing nice with keras fix: https://stackoverflow.com/questions/78233403/attributeerror-module-keras-tf-keras-keras-has-no-attribute-internal/78233592#78233592