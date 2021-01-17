import torch
from torch import nn
import math

"""
Все тензоры в задании имеют тип данных float32.
"""

class AE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.encoder = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.d)
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def encode(self, x):
        """
        Генерирует код по объектам.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: Tensor - матрица размера n x d.
        """
        return self.encoder(x)

    def decode(self, z):
        """
        По матрице латентных представлений z возвращает матрицу объектов x.
        Вход: z, Tensor - матрица n x d латентных представлений.
        Возвращаемое значение: Tensor, матрица объектов n x D.
        """
        return self.decoder(z)

    def batch_loss(self, batch):
        """
        Вычисляет функцию потерь по батчу - усреднение функции потерь
        по объектам батча.
        Функция потерь по объекту- сумма L2-ошибки восстановления по батчу и
        L2 регуляризации скрытых представлений с весом 1.
        Возвращаемое значение должно быть дифференцируемо по параметрам модели (!).
        Вход: batch, Tensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - функция потерь по батчу.
        """
        z = self.encode(batch)
        x_gen = self.decode(z)
        return (((batch - x_gen) ** 2).sum() + (z ** 2).sum()) / batch.shape[0]

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы объектов x. Использует стандартное нормальное
        распределение в пространстве представлений.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        z = torch.randn((num_samples, self.d), dtype=torch.float32)
        return self.decoder(z)


def log_mean_exp(data):
    """
    Возвращает логарифм среднего по последнему измерению от экспоненты данной матрицы.
    Подсказка: не забывайте про вычислительную стабильность!
    Вход: mtx, Tensor - тензор размера n_1 x n_2 x ... x n_K.
    Возвращаемое значение: Tensor, тензор размера n_1 x n_2 x ,,, x n_{K - 1}.
    """
    max_val, _ = torch.max(data, dim=-1, keepdim=True)
    return torch.log(torch.mean(torch.exp(data - max_val), dim=-1)) + max_val.squeeze(1)


def log_likelihood(x_true, x_distr):
    """
    Вычисляет логарфм правдоподобия объектов x_true для индуцированного
    моделью покомпонентного распределения Бернулли.
    Каждому объекту из x_true соответствуют K сэмплированных распределений
    на x из x_distr.
    Требуется вычислить оценку логарифма правдоподобия для каждого объекта.
    Подсказка: не забывайте про вычислительную стабильность!
    Подсказка: делить логарифм правдоподобия на число компонент объекта не надо.
    Вход: x_true, Tensor - матрица объектов размера n x D.
    Вход: x_distr, Tensor - тензор параметров распределений Бернулли
    размера n x K x D.
    Выход: Tensor, матрица размера n x K - оценки логарифма правдоподобия
    каждого сэмпла.
    """
    eps = 0.001
    eps_matrix = torch.ones(x_distr.shape) * eps
    cut = torch.min(torch.max(x_distr, eps_matrix), 1 - eps_matrix)
    return (x_true.unsqueeze(1) * torch.log(cut) +
            (1 - x_true).unsqueeze(1) * torch.log(1 - cut)).sum(dim=2)


def kl(q_distr, p_distr):
    """
    Вычисляется KL-дивергенция KL(q || p) между n парами гауссиан.
    Вход: q_distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: p_distr, tuple(Tensor, Tensor). Аналогично.
    Возвращаемое значение: Tensor, вектор размерности n, каждое значение которого - 
    - KL-дивергенция между соответствующей парой распределений.
    """
    p_mu, p_sigma = p_distr
    q_mu, q_sigma = q_distr
    ratio = (q_sigma / p_sigma) ** 2
    return 1 / 2 * (ratio - torch.log(ratio) + ((q_mu - p_mu) / p_sigma) ** 2 - 1).sum(dim=1)


class VAE(nn.Module):
    def __init__(self, d, D):
        """
        Инициализирует веса модели.
        Вход: d, int - размерность латентного пространства.
        Вход: D, int - размерность пространства объектов.
        """
        super(type(self), self).__init__()
        self.d = d
        self.D = D
        self.proposal_network = nn.Sequential(
            nn.Linear(self.D, 200),
            nn.LeakyReLU(),
        )
        self.proposal_mu_head = nn.Linear(200, self.d)
        self.proposal_sigma_head = nn.Linear(200, self.d)
        self.generative_network = nn.Sequential(
            nn.Linear(self.d, 200),
            nn.LeakyReLU(),
            nn.Linear(200, self.D),
            nn.Sigmoid()
        )

    def proposal_distr(self, x):
        """
        Генерирует предложное распределение на z.
        Подсказка: областью значений sigma должны быть положительные числа.
        Для этого при генерации sigma следует использовать softplus (!) в качестве
        последнего преобразования.
        Вход: x, Tensor - матрица размера n x D.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        soft_plus = nn.Softplus()
        prop = self.proposal_network(x)
        mu, sigma = self.proposal_mu_head(prop), self.proposal_sigma_head(prop)
        sigma = soft_plus(sigma)
        return mu, sigma

    def prior_distr(self, n):
        """
        Генерирует априорное распределение на z.
        Вход: n, int - число распределений.
        Возвращаемое значение: tuple(Tensor, Tensor),
        Каждый Tensor - матрица размера n x d.
        Первый - mu, второй - sigma.
        """
        mu = torch.zeros((n, self.d), dtype=torch.float32)
        sigma = torch.ones((n, self.d), dtype=torch.float32)
        return mu, sigma

    def sample_latent(self, distr, K=1):
        """
        Генерирует сэмплы из гауссовского распределения на z.
        Сэмплы должны быть дифференцируемы по параметрам распределения!
        Вход: distr, tuple(Tensor, Tensor). Каждое Tensor - матрица размера n x d.
        Первое - mu, второе - sigma.
        Вход: K, int - число сэмплов для каждого объекта.
        Возвращаемое значение: Tensor, матрица размера n x K x d.
        """
        mu, sigma = distr
        standart_distr = torch.randn((mu.shape[0], K, mu.shape[1]), dtype=torch.float32)
        return mu.unsqueeze(1) + sigma.unsqueeze(1) * standart_distr

    def generative_distr(self, z):
        """
        По матрице латентных представлений z возвращает матрицу параметров
        распределения Бернулли для сэмплирования объектов x.
        Вход: z, Tensor - тензор n x K x d латентных представлений.
        Возвращаемое значение: Tensor, тензор параметров распределения
        Бернулли размера n x K x D.
        """
        return self.generative_network(z)

    def batch_loss(self, batch):
        """
        Вычисляет вариационную нижнюю оценку логарифма правдоподобия по батчу.
        Вариационная нижняя оценка должна быть дифференцируема по параметрам модели (!),
        т. е. надо использовать репараметризацию.
        Требуется вернуть усреднение вариационных нижних оценок объектов батча.
        Вход: batch, FloatTensor - матрица объектов размера n x D.
        Возвращаемое значение: Tensor, скаляр - вариационная нижняя оценка логарифма
        правдоподобия по батчу.
        """
        q_dist = self.proposal_distr(batch)
        z = self.sample_latent(q_dist)
        x_gen = self.generative_distr(z)        
        log_like_batch = log_likelihood(batch, x_gen)
        kl_div_batch = kl(q_dist, self.prior_distr(batch.shape[0]))
        return (log_like_batch - kl_div_batch).mean()

    def generate_samples(self, num_samples):
        """
        Генерирует сэмплы из индуцируемого моделью распределения на объекты x.
        Вход: num_samples, int - число сэмплов, которые надо сгененрировать.
        Возвращаемое значение: Tensor, матрица размера num_samples x D.
        """
        prior_dist = self.prior_distr(1)
        samples = self.sample_latent(prior_dist, num_samples).squeeze(0)
        return self.generative_network(samples)


def gaussian_log_pdf(distr, samples):
    """
    Функция вычисляет логарифм плотности вероятности в точке относительно соответствующего
    нормального распределения, заданного покомпонентно своими средним и среднеквадратичным отклонением.
    Вход: distr, tuple(Tensor, Tensor). Каждый Tensor - матрица размера n x d.
    Первый - mu, второй - sigma.
    Вход: samples, Tensor - тензор размера n x K x d сэмплов в скрытом пространстве.
    Возвращаемое значение: Tensor, матрица размера n x K, каждый элемент которой - логарифм
    плотности вероятности точки относительно соответствующего распределения.
    """
    mu, sigma = distr
    first_part = - 1 / 2 * math.log(2 * math.pi) * sigma.shape[1] - torch.log(sigma).sum(dim=1, keepdim=True)
    log_exp = - ((samples - mu.unsqueeze(1)) ** 2  / (2 * sigma ** 2).unsqueeze(1)).sum(dim=2)
    return first_part + log_exp


def compute_log_likelihood_monte_carlo(batch, model, K):
    """
    Функция, оценку логарифма правдоподобия вероятностной модели по батчу методом Монте-Карло.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    prior_dist = model.prior_distr(batch.shape[0])
    z = model.sample_latent(prior_dist, K)
    x_gen = model.generative_distr(z)
    log_prob = log_likelihood(batch, x_gen)
    return float(log_mean_exp(log_prob).mean())


def compute_log_likelihood_iwae(batch, model, K):
    """
    Функция, оценку IWAE логарифма правдоподобия вероятностной модели по батчу.
    Оценка логарифма правдоподобия модели должна быть усреднена по всем объектам батча.
    Подсказка: не забудьте привести возращаемый ответ к типу float, иначе при вычислении
    суммы таких оценок будет строится вычислительный граф на них, что быстро приведет к заполнению
    всей доступной памяти.
    Вход: batch, FloatTensor - матрица размера n x D
    Вход: model, Module - объект, имеющий методы prior_distr, proposal_distr, sample_latent и generative_distr,
    описанные в VAE.
    Вход: K, int - количество сэмплов.
    Возвращаемое значение: float - оценка логарифма правдоподобия.
    """
    proposal_dist = model.proposal_distr(batch)
    z = model.sample_latent(proposal_dist, K)
    x_gen = model.generative_distr(z)
    prior_dist = model.prior_distr(batch.shape[0])
    log_prob = log_likelihood(batch, x_gen) + gaussian_log_pdf(prior_dist, z) - gaussian_log_pdf(proposal_dist, z)
    return float(log_mean_exp(log_prob).mean())
