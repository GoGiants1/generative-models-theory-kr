# DART(Denoising AutoRegressive Transformer)

## Introduction

In parallel, autoregressive models, such as **GPT-4** (Achiam et al., 2023), have shown great success in modeling long-range dependencies in sequential data, particularly in the field of natural language processing.

These models **efficiently cache computations and manage dependencies across time steps**, which has also inspired research into adapting autoregressive models for image generation.

- 초기 연구 픽셀 CNN은 높은 계산 비용이 들었음(그 이유는 픽셀 공간에서 수행하기 때문임)
- VQ-GAN, Parti, Chameleon, MAR 같은 최신 모델들은 퀀타이즈드 된 latent 스페이스에서 모델링을 한다
  - 하지만 이 모델들은 디퓨전 기반의 로스함수를 사용함. these models fail to fully leverage the progressive denoising benefits, resulting in limited global context and error propagation during generation

이 페이퍼에서 제시한 모델은

- non-Markovian diffusion framework(DDIM)을 활용한다
  - enables the model to leverage the full generative trajectory during training and inference, while retaining the progressive modeling benefits of diffusion models, resulting in more efficient and flexible generation compared to traditional diffusion and autoregressive models.
- DDIM(논 마코비언)의 단점을 극복하기 위해서 두 가지 접근법을 제시함.
    1. 토큰 수준의 오토리그레시브 모델링
    2. 플로우기반의 수정 모듈

연구에서 제시한 모델은 class-conditioned, text conditioned image generation 태스크에서 기존 모델들과 견줄만한 성능을 보여주고, scalable, unified approach를 제시함.

## Background

### Diffusion Models

이미지 $x_0 \in \mathbb{R}^{3 \times H \times W}$가 주어졌을때, 일련의 latent variable $x_t$를 원래의 이미지 $x_0$에서 노이즈를 점진적으로 더하는 마르코브 과정을 통해 정의할 수 있다. 이 과정에 사용되는 transition $q(x_t|x_{t-1})$ , marginal $q(x_t|x_0)$는 다음과 같이 정의된다.

$$
q(\bm{x_t}|\bm{x_{t-1}}) = \mathcal{N}(\bm{x_t}; \sqrt{1- \beta_t}\bm{x_{t-1}}, \beta_t \mathbf{I})
$$

$$
q(\bm{x_t}|\bm{x_0}) = \mathcal{N}(\bm{x_t}; \sqrt{\bar{\alpha}_t}\bm{x_0}, (1- \bar{\alpha}_t) \mathbf{I})
$$

where $\bar{\alpha}_t = \prod_{\tau=1}^{t} (1-\beta_{\tau})$, $0 < \beta_t < 1$는 노이즈 스케줄에 따라 결정된다.

모델은 이러한 디퓨전 프로세스의 reverse process를 통해 이미지의 노이즈를 제거하고, 이미지를 생성한다. 이때, 사용되는 reverse model을 배우는 것이 목표이며, training objective는 다음과 같이 정의된다.

$$
min \mathcal{L}_{\theta}^{\text{DM}} = \mathbb{E}_{t \sim [1, T],\bm{x_t} \sim q(\bm{x_t}|\bm{x_0})} [\omega_t \cdot \parallel \bm{x_\theta}(\bm{x_t}, t) - \bm{x_0} ]
$$

이 수식에서 $\bm{x_\theta}(\bm{x_t}, t)$는 time-conditioned denoiser를 의미하며 노이지한 sample $x_t$와 깨끗한 원본 이미지 $x_0$의 매핑을 학습한다. $\omega_t$는 시간에 따라 변하는 loss weight이다. 주로 SNR이나 SNR+1을 사용한다. 더 효율적인 학습을 위해 $x_t$에 reparameterization trick을 적용하여 noise or v-prediction 형식으로 변환할 수 있다. 그리고 픽셀 space or latent space에서 denoiser를 학습할 수 있다.

이러한 보편적인 디퓨전 기반의 모델들은 연산량이 많고 비효율적이다. inference time에는 많은 디노이징 스텝이 필요하며, 많은 학습데이터를 필요로 한다. 더 나아가, 이러한 모델들은 generation context 정보를 효율적으로 활용할 수 있는 능력이 없어서, 복잡한 이미지나 비디오와 같은 긴 시퀀스를 생성하는데 한계가 있다.


### Autoregressive Models

Natural language processing에서 트랜스포머와 같은 autoregressive 모델들은 텍스트 시퀀스의 긴 의존성을 모델링하는데 효과적이어서 큰 성공을 거두었다. 이 모델링 방법은 이미지 생성 연구에서도 적용되고 있다. 디퓨전 기반 모델과는 달리 autoregressive 기반의 이미지 생성 모델은 discrete image tokens(Vector Quantized tokens) 간의 관계를 배우는데에 집중한다.

전반적인 과정을 설명하자면 다음과 같다.
이미지 $x \in \mathbb{R}^{3 \times H \times W}$ 를 이미지 인코더 $\mathcal{E}$를 통해 latent token sequence $z_{1:\bm{N}} = \mathcal{E}(x)$로 변환한다. 이 discrete 토큰은 디코더 $\mathcal{D}$를 통해, $\hat{x}= \mathcal{D}(z_{1:\bm{N}})$로 원래 이미지와 유사하게 복원된다($\hat{x} \approx x$).

Autoregressive 모델은 $z_n$과 이전 토큰들 $z_{1:n-1}$ 사이의 조건부 확률을 모델링한다 (**next token prediction**). 그래서 AutoRegressive 모델의 training objective는 다음과 같이 정의된다.

$$
\text{max} \ \mathcal{L}_{\theta}^{\text{CE}} = \text{max} \ \Sigma_{n=1}^{N} \log P_{\theta}(z_n|z_{0:n-1})
$$




