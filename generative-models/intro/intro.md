# Generative Modeling: An Introduction
생성 모델은 joint probability distribution을 학습하여 새로운 데이터를 생성하는 모델입니다. 이러한 모델은 이미지, 텍스트, 음악 등 다양한 데이터 유형을 생성할 수 있으며, GANs, VAEs, autoregressive models, normalizing flows, diffusion models, energy based models(EBM) 등 다양한 종류가 있습니다. 이 글은 `Probabilistic Machine Learning: Advanced Topics(2023)` 책을 공부하며 작성한 글입니다.

## What is Generative Modeling?
- 생성 모델은 joint probability distribution $p(x)$를 학습합니다. 이때, $x \in \mathcal{X}$는 데이터 포인트를 나타냅니다.
- 몇몇 생성 모델은 input이나 covariates $c \in \mathcal{C}$를 사용하여 conditional generative $p(x|c)$를 학습합니다.

## Types of Generative Models
생성 모델을 분류할 때, 다음과 같은 기준을 사용할 수 있습니다.

- **Density**: 모델이 pdf 함수 $p(\mathbb{x})$ pointwise evaluation을 지원하는지, 이 pdf 함수가 정확한지, lower bound(or 근사값)인지, 빠르거나 느린지 등으로 구별할 수 있습니다. Implicit density models는 확률 분포를 직접적으로 모델링하지 않습니다.(e.g., GANs, )

- **Sampling**: 모델이 새로운 데이터 포인트 $x \sim p(x)$를 생성할 수 있는지, 생성 속도, 샘플링 정확도(exact sampling vs. approximate sampling) 등으로 구별할 수 있습니다.

- **Training**: 모델이 어떻게 파라미터 추정에 사용하는 방법에 따라 구별할 수 있습니다. 어떤 모델의 경우 maximum likelihood estimation(MLE)을 사용할 수 있습니다. 반면에, objective function이 non-convex인 경우, 오직 local optimum에 수렴할 수 있습니다. VAE의 경우 ELBO를 최대화하는 방법을 사용합니다. GANs의 경우 adversarial training을 통하여 min-max optimization을 수행하고 이러한 이유로 학습이 불안정할 수 있습니다.

- **Latents**: 모델이 latent variables $z$를 사용하는지 여부로 구별할 수 있습니다. 그리고, 세부적으로 latent가 compressed representation 인가로 구별할 수 있습니다. VAEs, GANs, autoregressive models, diffusion model은 latent variables을 사용합니다

- **Architecture**: 모델이 어떤 구조를 사용하는지로 구별할 수 있습니다. 그리고 restriction이 있는지로 구별할 수 있습니다. 예를 들어, Flow-based models는 invertible neural networks(tractable한 Jacobian을 가져야 함)를 사용해야 한다는 제약이 있습니다.

## Goals of Generative Modeling

- **Data Generation**: 새로운 데이터를 생성하는 것이 목표입니다. 이는 이미지, 텍스트, 음악 등 다양한 데이터 유형을 생성할 수 있습니다. 그리고 이때 conditional generative model $p(x|c)$를 사용하면 특정한 조건에 맞는 데이터를 생성할 수 있습니다.

- **Density Estimation**: 관찰된 데이터 벡터의 확률을 추정하는 것이 목표입니다. 이는 데이터의 likelihood를 계산할 수 있게 해줍니다. 이는 다양한 task에 사용될 수 있습니다. 예를 들어, outlier detection, data compression, generative classifier, model comparison 등에 사용될 수 있습니다. 가장 간단한 방법으로는 KDE(Kernel Density Estimation)을 사용할 수 있습니다. 이 방법은 데이터를 kernel function으로 smoothing하여 pdf를 추정합니다. 낮은 차원에서는 잘 작동하지만, 고차원에서는 잘 작동하지 않습니다(차원의 저주).

- **Imputation**: 누락된 데이터를 채우는 것이 목표입니다. 이는 데이터의 missing value를 채우는 것을 의미합니다. 생성 모델을 이용하여 이 task를 일반화 할 수 있습니다. 관찰된 데이터 $p(\mathbb{x}_o)$에 생성 모델을 fitting 시키고 새로운 데이터 $p(\mathbb{X}_m| \mathbb{X}_o)$를 샘플링하여 누락된 데이터를 채울 수 있습니다. 이를 **multiple imputation**이라고 하며, 이미지에 적용될 경우 가려진 이미지 부분을 채울 수 있습니다(**in-painting**).

- **Structure Discovery**: 데이터의 저차원 구조(low-dim pattern, latent)를 발견하는 것이 목표입니다. 베이즈 법칙을 사용하여 $p(z|x) \propto p(x|z)p(z)$를 계산하여 데이터의 구조를 발견할 수 있습니다. 이는 데이터의 manifold를 학습하는 것을 의미합니다. 이러한 방법은 데이터의 차원을 줄이는데 사용될 수 있습니다. 예를 들어, VAEs는 데이터의 manifold를 학습하여 데이터의 차원을 줄일 수 있습니다. Latent Diffusion Model은 VAE가 학습한 manifold를 활용하여 효율적으로 데이터를 생성합니다.

- **Latent space interpolation**: latent space에서 두 점 사이의 interpolation을 수행하는 것이 목표입니다. 이는 latent space에서 두 점 사이의 linear interpolation($z = \lambda z_1 + (1 - \lambda) z_2$, where $0 \le \lambda \le 1$)을 수행하여 새로운 데이터를 생성할 수 있습니다. 예를 들어, VAEs는 latent space에서 smooth한 interpolation을 보여줍니다.

## References

```bibtex
@book{pml2Book,
 author = "Kevin P. Murphy",
 title = "Probabilistic Machine Learning: Advanced Topics",
 publisher = "MIT Press",
 year = 2023,
 url = "http://probml.github.io/book2"
}
```