# 디퓨전 모델 배우기: VAE부터 DDPM, SDE, CFG까지

최근에 연구실에서 신입생 위주로 디퓨전 모델에 대한 스터디를 진행하고 있다. [Generative Modeling by Estimating Gradients of the Data Distribution](https://arxiv.org/abs/1907.05600)을 시작으로, [Score-Based Generative Modeling through Stochastic Differential Equations](https://arxiv.org/abs/2011.13456) 논문을 읽었고, 이제 [DDPM](https://arxiv.org/abs/2006.11239)을 공부할 차례가 되었다. 하지만, 앞선 논문들을 읽을 때 높은 난이도 및 요구되는 선행 지식이 많아 이해에 어려움을 겪었다. 따라서 개별적인 논문을 각각 읽는 것은 현 시점에서는 비효율적이라 판단하여, 이를 한번에 다루는 [A Unified Perspective](https://arxiv.org/abs/2208.11970)를 읽기로 했다. 이 논문은 디퓨전 모델의 개념을 이해하는데 필요한 배경 지식을 제공하고, VAE부터 VDM, DDPM, SDE, CFG까지 다양한 디퓨전 모델을 통합적으로 이해할 수 있도록 도와준다.

이 논문을 읽은 뒤, 핵심 내용 정리 및 복습을 위해 이 글을 작성하였다.

## Background: ELBO, VAE, and Hierarchical  VAE

여러 모달리티에서, 우리는 관측된 데이터를 우리가 알지 못하는 임의의 latent $z$로 표현되거나, latent로부터 생성된 것으로 생각할 수 있다. 여기서는 플라톤의 동굴의 우화를 활용하여 이러한 개념을 설명하고 있다. 이 우화에서는 동굴 속의 사람들은 동굴 속의 2차원 그림자(observation)만을 보고 살아가지만, 사실 그 그림자는 더욱 복잡한 3차원 세계(latent)가 투영된 결과물이다. 이러한 관점에서, 우리는 관측된 데이터 $x$를 생성하는 데 사용되는 고차원의 latent $z$를 찾는 것이 목표이다.

하지만, Generative modeling에서는 고차원의 latent를 찾는 것이 아닌, 저차원의 latent representation을 찾는 것을 목표로 한다. 이는 데이터의 차원이 높아질수록 데이터의 분포를 학습하기 어려워지기 때문이다. 따라서, 저차원의 latent representation을 찾아 데이터의 분포를 학습하고, 이를 통해 새로운 데이터를 생성하는 것이 목표이다.

### Basics: Conditional Probability & Chain Rule

**Conditional Probability**: 두 사건 A, B에 대한 조건부 확률은 다음과 같이 정의된다. Conditional probability A given B는 B가 발생했을 때, A가 발생할 확률을 의미한다.

$$
P(A|B) = P_B(A) = \frac{P(A, B)}{P(B)}
$$

**Chain Rule**: 두 사건 A, B에 대한 intersection을 표현하는 방법으로 chain rule을 사용할 수 있다.

$$
P(A \cap B) = P(A, B) = P(A|B)P(B) = P(B|A)P(A)
$$

### Basics: Bayes Rule

이 글을 읽기 위해서는 몇가지 확률적인 정의에 대한 복습이 필요하다. 먼저, Bayes Rule은 다음과 같이 정의된다.

$$
p(z|x) = \frac{p(x|z)p(z)}{p(x)}
$$

- $p(z|x)$: posterior distribution, 관측된 데이터 $x$가 주어졌을 때, latent variable $z$의 확률 분포 (우리가 알고 싶은 것, 즉 주어진 데이터 $x$로부터 latent variable $z$를 추정하기 위한 분포)
- $p(x|z)$: likelihood, latent variable $z$가 주어졌을 때, 데이터 $x$의 확률 분포 (관측 데이터 $x$가 latent variable $z$에 의해 설명될 가능성)
- $p(z)$: prior distribution, latent variable $z$의 확률 분포(우리가 사전적으로 가지고 있는 $z$에 대한 지식이나 가정, 사전 확률 분포)
- $p(x)$: evidence, 데이터 $x$의 확률 분포(모든 가능한 latent variable $z$에 대해 $p(x|z)p(z)$의 총합으로 계산되는 값, joint distribution $p(x,z)$에서 latent $z$를 marginalize한 값)

#### Bayes Rule with 3 Variables

이 논문에서는 수식 전개를 자세히 진행하는데, 그 과정 속에서 Bayes Rule을 3개의 변수에 대해 확장하여 설명하고 있다. 자세한 과정은 설명되어 있지 않아서 이해하기 어려웠다. 하지만, 차근차근 수식을 전개해보면 다음과 같다.

수식 (46)을 도출해내는 과정을 살펴보면,

$$
q(x_t|x_{t-1}, x_0) = \frac{q(x_t, x_{t-1}, x_0)}{q(x_{t-1}, x_0)}\\
= \frac{q(x_{t-1}|x_{t},x_0)q(x_{t}, x_0)}{q(x_{t-1}, x_0)}\\= \frac{q(x_{t-1}|x_{t},x_0)q(x_{t}| x_0)\cancel{q(x_0)}}{q(x_{t-1}| x_0)\cancel{q(x_0)}}\\
$$

위와 같으며, 조건부 확률의 정의와 chain rule을 순차적으로 이용한다.

### Evidence Lower Bound (ELBO)

수학적으로, latent variable $z$와 우리가 관착한 data $x$를 joint distribution $p(x, z)$로 표현할 수 있다. 이를 marginalize하여 observed data의 likelihood $p(x)$를 구하려고 시도할 수 있다.

$$
p(x) = \int p(x, z) dz
$$

또는 chain rule of probability를 이용하여 다음과 같이 표현할 수 있다.

$$
p(x) =  p(x|z)p(z)
$$

하지만, 이를 계산하고 likelihood를 최대화 하는 것은 모든 latent variable $z$에 대해 marginalize하는 것이 필요하므로 복잡한 모델에서는 intractable하다. 따라서, 이를 근사하기 위해 위의 두 가지 $p(x)$ 식을 이용하여 ELBO 수식을 유도할 수 있다. 먼저, Evidence란 우리가 관측한 데이터가 주어졌을 때, 모델이 이 데이터를 생성할 확률을 의미한다. 여기서는 log를 취한 형식을 사용한다.

$$
\log p(x) \ge \mathbb{E}_{q_{\phi}(z|x)} \left [ \frac{\log p(x, z)}{ \log q_{\phi}(z|x)} \right ]  = \text{ELBO}
$$

여기서 $q_{\phi}(z|x)$는 flexible approximate variational distribution을 의미한다.

이 ELBO 수식을 유도하는 방법은 두 가지가 있다. 하나는 Jensen's inequality를 이용하는 방법이고, 다른 하나는 KL divergence를 이용하는 방법이다. 전자는 우리에게 유용한 정보를 제공하지 못하기 때문에 후자를 이용하여 ELBO를 유도하는 것이 더욱 많은 인사이트를 제공한고 한다.

유도하는 수식을 살펴보면 아래와 같다. 이때, trick으로는 $1 = \smallint q_{\phi}(z|x) dz = \frac{q_{\phi}(z|x)}{q_{\phi}(z|x)}$와 평균의 정의 등을 이용한다.

$$
\log p(x) = 
\log p(x) \smallint q_{\phi}(z|x) dz \\
=\smallint q_{\phi}(z|x) \log p(x) dz 
\\= \mathbb{E}_{q_{\phi}(z|x)} \log \frac{p(x, z)}{p(z|x)} dz
\\=\mathbb{E}_{q_{\phi}(z|x)} \left [ \log \frac{p(x, z)}{q_{\phi}(z|x)} \right] + \mathbb{E}_{q_{\phi}(z|x)}\left[ \log \frac{q_{\phi}(z|x)}{p(z|x)} \right] = \\
\smallint q_{\phi}(z|x) \log \frac{p(x, z)}{q_{\phi}(z|x)} dz + D_{KL}(q_{\phi}(z|x) || p(z|x))
$$

이때, $D_{KL}(q_{\phi}(z|x) || p(z|x))$ 는 KL divergence로, 두 확률 분포 사이의 거리를 측정하는 지표이다. 이 값은 항상 0보다 크거나 같다는 점을 이용하면, ELBO에 대한 이해를 높일 수 있다.

먼저, 이 식에서

- $q_{\phi}(z|x)$: approximate posterior distribution
- $p(z|x)$: true posterior distribution
임을 알아두자.

1. Evidence - ELBO = KL divergence에서 KL divergence는 항상 0보다 크거나 같기 때문에, ELBO는 evidence보다 작거나 같다.
2. 우리의 evidence($\log(p(x))$)는 $\phi$에 관해서는 상수이기 때문에,ELBO + KL Divergence = constant 이다. 따라서, ELBO를 $\phi$에 대해 최대화하는 것은 KL divergence를 최소화하는 것과 같다. 이렇게 간접적으로 KL divergence를 최소화하는 것이 목표이다.(직접 최적화 하지 못하는 이유는 KL divergence는 두 확률 분포 사이의 거리를 측정하는 지표이나, true posterior $p(z|x)$에 우리가 접근할 수 없기에 intractable하기 때문이다.)

즉, ELBO는 true latent posterior를 배우기 위한 proxy인 것이다.

이는 Variational Inference 방법론에서 부터 시작된 개념이며, 이를 설명하는 그림을 [pml-book2](http://probml.github.io/book2)에서 가져왔다.


![Variational inference](https://github.com/user-attachments/assets/5983394b-772e-4c27-9351-6a4e9345d3cf)

### Variational Autoencoder (VAE)

### Hierarchical VAE