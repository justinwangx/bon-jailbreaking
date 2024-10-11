import numpy as np
import scipy
from scipy.optimize import curve_fit, least_squares
from tqdm import tqdm

from almj.utils.utils import file_cache


def huber_loss(diffs, delta=1e-3):
    # Huber loss. no reduction.
    # see https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html#torch.nn.HuberLoss

    # intuition: L1 loss when we have an outlier point (diff > delta), L2 loss for all elsewhere
    return np.where(
        np.abs(diffs) < delta,
        0.5 * diffs**2,
        delta * (np.abs(diffs) - 0.5 * delta),
    )


def loss(fit_params, x, y):
    assert len(fit_params) == 3, "unexpected number of scaling law params to fit"
    alpha, e, a = fit_params
    # alpha: scaling exponent
    # e: error term
    # a: intercept term
    # x: parameters (steps in our case)
    # This function models the power law relationship between the number of parameters (x) and the loss (y).
    # The power law is represented as: Loss = A * x^(-alpha) + E = (A / x^alpha) + E
    # This can be rewritten as: log(Loss) = log(A) - alpha * log(x) + log(E)
    # Since log(A) and log(E) are constants, we can represent this as: log(Loss) = a - alpha * log(x) + e
    # We then take the logsumexp of the terms a - alpha * log(x) and e to stabilize the computation.
    # We use the Huber loss to handle outliers and ensure robust fitting.
    # Huber_{delta}(LSE(a - alpha log(x_i), e) - log(Loss_i) ))
    # See https://arxiv.org/pdf/2203.15556 P.25 (Appendix D.2)
    concatted = np.stack(
        [a - alpha * np.log(x), np.broadcast_to(np.array([e]), shape=(x.shape[0],))]  # manual broadcast. annoying.
    )
    # LSE: Log-Sum-Exp of the concatenated array, which stabilizes the computation of the log of the sum of exponentials
    LSE = scipy.special.logsumexp(concatted, axis=0)

    # apply huber loss
    losses = huber_loss(LSE - np.log(y))

    # reduce losses over all runs
    return np.sum(losses)


## Chinchilla paper's described methodology
## referenced and built off of: Chinchilla paper ; Epoch AI code (https://github.com/epoch-research/analyzing-chinchilla/blob/main/data_analysis.ipynb)
@file_cache()
def chinchilla_fit(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    inps = (x, y)

    best_loss = np.inf
    best_result = None

    for alpha in tqdm(np.arange(0, 2.5, 0.5)):
        for e in np.arange(-2.0, 1.5, 0.5):
            for a in np.arange(0, 30, 5):
                result = scipy.optimize.minimize(loss, x0=[alpha, e, a], args=inps, method="L-BFGS-B")

                if result.fun < best_loss:
                    best_result = result.x
                    best_loss = result.fun
                    # print(f"New best loss: {best_loss}")
                    # print(f"New best params: {best_result}")
                    # print(f"From starting condition: {[alpha, e, a]}")

    (alpha, e, a) = best_result

    a = np.exp(a)
    b = alpha
    c = np.exp(e)
    return a, b, c


def power_law(x, a, b, c):
    return a * np.power(x, -b) + c


def exp_power_law(x, a, b, c):
    return np.exp(-power_law(x, a, b, c))


@file_cache()
def simple_fit(x, y, fit_raw_data=True, epsilon=1e-5):
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
        function = power_law
    else:
        y = y.mean(axis=0)
        function = exp_power_law
    params, _ = curve_fit(function, x, y, p0=[1, 0.5, 0.1], bounds=(0, np.inf), maxfev=10000)
    a = params[0]
    b = params[1]
    c = params[2]
    return a, b, c


def loss_two_terms(fit_params, x, y):
    assert len(fit_params) == 4, "Unexpected number of scaling law params to fit"
    a_log, b, c_log, d = fit_params
    # No scalar term here, so no need to broadcast
    concatted = np.stack([a_log - b * np.log(x), c_log - d * np.log(x)])
    LSE = scipy.special.logsumexp(concatted, axis=0)
    losses = huber_loss(LSE - np.log(y))
    return np.sum(losses)


def loss_two_terms_plus_e(fit_params, x, y):
    assert len(fit_params) == 5, "Unexpected number of scaling law params to fit"
    a_log, b, c_log, d, e_log = fit_params
    e_array = np.broadcast_to(e_log, x.shape)
    concatted = np.stack([a_log - b * np.log(x), c_log - d * np.log(x), e_array])
    LSE = scipy.special.logsumexp(concatted, axis=0)
    losses = huber_loss(LSE - np.log(y))
    return np.sum(losses)


@file_cache()
def chinchilla_fit_two_terms(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating chinchilla_fit_two_terms")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for b in tqdm(np.arange(0, 2.5, 0.5)):
        for d in np.arange(0, 2.5, 0.5):
            for a_log in np.arange(-5, 5, 1):
                for c_log in np.arange(-5, 5, 1):
                    initial_guess = [a_log, b, c_log, d]
                    result = scipy.optimize.minimize(loss_two_terms, x0=initial_guess, args=inps, method="L-BFGS-B")
                    if result.fun < best_loss:
                        best_result = result.x
                        best_loss = result.fun

    a_log, b, c_log, d = best_result
    a = np.exp(a_log)
    c = np.exp(c_log)
    return a, b, c, d


@file_cache()
def chinchilla_fit_two_terms_plus_e(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating chinchilla_fit_two_terms_plus_e")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for b in tqdm(np.arange(0, 2.5, 0.5)):
        for d in np.arange(0, 2.5, 0.5):
            for a_log in np.arange(-5, 5, 1):
                for c_log in np.arange(-5, 5, 1):
                    for e_log in np.arange(-5, 5, 1):
                        initial_guess = [a_log, b, c_log, d, e_log]
                        result = scipy.optimize.minimize(
                            loss_two_terms_plus_e, x0=initial_guess, args=inps, method="L-BFGS-B"
                        )
                        if result.fun < best_loss:
                            best_result = result.x
                            best_loss = result.fun

    a_log, b, c_log, d, e_log = best_result
    a = np.exp(a_log)
    c = np.exp(c_log)
    e = np.exp(e_log)
    return a, b, c, d, e


def power_law_two_terms(x, a, b, c, d):
    return a * np.power(x, -b) + c * np.power(x, -d)


def power_law_two_terms_plus_e(x, a, b, c, d, e):
    return a * np.power(x, -b) + c * np.power(x, -d) + e


def exp_power_law_two_terms(x, a, b, c, d):
    return np.exp(-power_law_two_terms(x, a, b, c, d))


def exp_power_law_two_terms_plus_e(x, a, b, c, d, e):
    return np.exp(-power_law_two_terms_plus_e(x, a, b, c, d, e))


@file_cache()
def simple_fit_two_terms(x, y, fit_raw_data=True, epsilon=1e-5):
    print("Calculating simple_fit_two_terms")
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
        function = power_law_two_terms
    else:
        y = y.mean(axis=0)
        function = exp_power_law_two_terms
    params, _ = curve_fit(
        function,
        x,
        y,
        p0=[1, 0.5, 1, 0.5],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    a, b, c, d = params
    return a, b, c, d


@file_cache()
def simple_fit_two_terms_plus_e(x, y, fit_raw_data=True, epsilon=1e-5):
    print("Calculating simple_fit_two_terms_plus_e")
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
        function = power_law_two_terms_plus_e
    else:
        y = y.mean(axis=0)
        function = exp_power_law_two_terms_plus_e
    params, _ = curve_fit(
        function,
        x,
        y,
        p0=[1, 0.5, 1, 0.5, 0.1],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    a, b, c, d, e = params
    return a, b, c, d, e


def power_law_single_term_no_constant(x, a, b):
    # Single-term power-law model without a constant term.
    return a * np.power(x, -b)


def exp_power_law_single_term_no_constant(x, a, b):
    # Single-term exponential power-law model without a constant term.
    return np.exp(-power_law_single_term_no_constant(x, a, b))


def loss_single_term(fit_params, x, y):
    # Loss function for the single-term power-law model.
    # Uses Huber loss to be robust against outliers.
    a_log, b = fit_params
    y_pred_log = a_log + b * np.log(x)
    y_actual_log = np.log(y)
    diffs = y_pred_log - y_actual_log
    losses = huber_loss(diffs)
    return np.sum(losses)


def chinchilla_fit_single_term_no_constant(x: np.ndarray, y: np.ndarray):
    # Fitting function using optimization with Huber loss.
    # Searches over a grid of initial parameters for robustness.
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating chinchilla_fit_single_term_no_constant")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for b in tqdm(np.linspace(-2, 2, 20)):
        for a_log in np.linspace(-5, 5, 20):
            initial_guess = [a_log, b]
            result = scipy.optimize.minimize(loss_single_term, x0=initial_guess, args=inps, method="L-BFGS-B")
            if result.fun < best_loss:
                best_result = result.x
                best_loss = result.fun

    a_log, b = best_result
    a = np.exp(a_log)
    return a, b


def simple_fit_single_term_no_constant(x, y, fit_raw_data=True, epsilon=1e-5):
    # Simple fitting function that can fit raw data or log-transformed data.
    print("Calculating simple_fit_single_term_no_constant")
    if not fit_raw_data:
        # Transform data to log-log space
        x_log = np.log(x)
        y_log = np.log(y + epsilon)
        y_mean = y_log.mean(axis=0) if y_log.ndim > 1 else y_log

        def linear_model(x_log, a_log, b):
            return a_log + b * x_log

        function = linear_model
        initial_guess = [0.0, 0.0]  # Initial guess for log(a) and b
        params, _ = curve_fit(
            function,
            x_log,
            y_mean,
            p0=initial_guess,
            maxfev=10000,
        )
        a_log, b = params
        a = np.exp(a_log)
    else:
        y_mean = y.mean(axis=0) if y.ndim > 1 else y
        function = power_law_single_term_no_constant
        initial_guess = [1.0, 0.5]
        params, _ = curve_fit(
            function,
            x,
            y_mean,
            p0=initial_guess,
            bounds=(0, np.inf),
            maxfev=10000,
        )
        a, b = params
    return a, b


def exp_decay_model(x, a, b, c):
    return a * np.exp(-b * x) + c


def exp_exp_decay_model(x, a, b, c):
    return np.exp(-exp_decay_model(x, a, b, c))


def loss_exp_decay(fit_params, x, y):
    assert len(fit_params) == 3, "Unexpected number of scaling law params to fit"
    a, b, c = fit_params
    y_pred = exp_decay_model(x, a, b, c)
    diffs = y_pred - y
    losses = huber_loss(diffs)
    return np.sum(losses)


@file_cache()
def chinchilla_fit_exp_decay(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating fit_exp_decay")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for a in tqdm(np.linspace(0.1, 10, 20)):
        for b in np.linspace(0.1, 10, 20):
            for c in np.linspace(0.1, 10, 20):
                initial_guess = [a, b, c]
                result = scipy.optimize.minimize(loss_exp_decay, x0=initial_guess, args=inps, method="L-BFGS-B")
                if result.fun < best_loss:
                    best_result = result.x
                    best_loss = result.fun

    a, b, c = best_result
    return a, b, c


@file_cache()
def simple_fit_exp_decay(x, y, fit_raw_data=True, epsilon=1e-5):
    print("Calculating simple_fit_exp_decay")
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
    else:
        y = y.mean(axis=0)
    function = exp_decay_model
    params, _ = curve_fit(
        function,
        x,
        y,
        p0=[1, 0.5, 0.1],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    a, b, c = params
    return a, b, c


def exp_poly_model(x, a, b, c, d, e):
    return a * np.exp(-b * x) + c * np.power(x, -d) + e


def exp_exp_poly_model(x, a, b, c, d, e):
    return np.exp(-exp_poly_model(x, a, b, c, d, e))


def loss_exp_poly(fit_params, x, y):
    assert len(fit_params) == 5, "Unexpected number of scaling law params to fit"
    a, b, c, d, e = fit_params
    y_pred = exp_poly_model(x, a, b, c, d, e)
    diffs = y_pred - y
    losses = huber_loss(diffs)
    return np.sum(losses)


@file_cache()
def chinchilla_fit_exp_poly(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating chinchilla_fit_exp_poly")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for a in tqdm(np.linspace(0.1, 10, 20)):
        for b in np.linspace(0.1, 10, 20):
            for c in np.linspace(0.1, 10, 20):
                for d in np.linspace(0.1, 10, 20):
                    for e in np.linspace(0.1, 10, 20):
                        initial_guess = [a, b, c, d, e]
                        result = scipy.optimize.minimize(loss_exp_poly, x0=initial_guess, args=inps, method="L-BFGS-B")
                        if result.fun < best_loss:
                            best_result = result.x
                            best_loss = result.fun

    a, b, c, d, e = best_result
    return a, b, c, d, e


@file_cache()
def simple_fit_exp_poly(x, y, fit_raw_data=True, epsilon=1e-5):
    print("Calculating simple_fit_exp_poly")
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
    else:
        y = y.mean(axis=0)
    function = exp_poly_model
    params, _ = curve_fit(
        function,
        x,
        y,
        p0=[1, 0.5, 1, 0.5, 0.1],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    a, b, c, d, e = params
    return a, b, c, d, e


def functional_form(x, a, b, c, d):
    return a * (1 + x / b) ** -c + d


def exp_functional_form(x, a, b, c, d):
    return np.exp(-functional_form(x, a, b, c, d))


def loss_functional_form(fit_params, x, y):
    assert len(fit_params) == 4, "Unexpected number of scaling law params to fit"
    a, b, c, d = fit_params
    y_pred = functional_form(x, a, b, c, d)
    diffs = y_pred - y
    losses = huber_loss(diffs)
    return np.sum(losses)


@file_cache()
def chinchilla_fit_functional_form(x: np.ndarray, y: np.ndarray):
    assert x.shape == y.shape, "x and y must have the same shape"
    print("Calculating chinchilla_fit_functional_form")
    inps = (x, y)
    best_loss = np.inf
    best_result = None

    for a in tqdm(np.linspace(0.1, 10, 20)):
        for b in np.linspace(0.1, 10, 20):
            for c in np.linspace(0.1, 10, 20):
                for d in np.linspace(0.1, 10, 20):
                    initial_guess = [a, b, c, d]
                    result = scipy.optimize.minimize(
                        loss_functional_form, x0=initial_guess, args=inps, method="L-BFGS-B"
                    )
                    if result.fun < best_loss:
                        best_result = result.x
                        best_loss = result.fun

    a, b, c, d = best_result
    return a, b, c, d


@file_cache()
def simple_fit_functional_form(x, y, fit_raw_data=True, epsilon=1e-5):
    print("Calculating simple_fit_functional_form")
    if not fit_raw_data:
        y = -np.log(y + epsilon)
        y = y.mean(axis=0)
        function = functional_form
    else:
        y = y.mean(axis=0)
        function = exp_functional_form
    params, _ = curve_fit(
        function,
        x,
        y,
        p0=[1, 1, 1, 0.1],
        bounds=(0, np.inf),
        maxfev=10000,
    )
    a, b, c, d = params
    return a, b, c, d


# Assuming the necessary model functions are already defined:
# power_law, power_law_two_terms, power_law_two_terms_plus_e,
# power_law_single_term_no_constant, exp_decay_model, exp_poly_model,
# functional_form, and their corresponding exponential versions.


@file_cache()
def o1_simple_fit_power_law(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the power law model using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = power_law
    else:
        y_transformed = y
        function = exp_power_law

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5, 0.1],
        [np.max(y_mean), 1.0, np.min(y_mean)],
        [0.1, 1.0, 0.0],
        [10.0, 0.5, 0.1],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), loss="huber", max_nfev=10000
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c = best_params
    return a, b, c


@file_cache()
def o1_simple_fit_power_law_two_terms(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the two-term power law model using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = power_law_two_terms
    else:
        y_transformed = y
        function = exp_power_law_two_terms

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5, 1.0, 0.5],
        [np.max(y_mean), 1.0, np.max(y_mean) / 2, 1.0],
        [0.1, 0.5, 0.1, 0.5],
        [10.0, 1.0, 5.0, 0.5],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), loss="huber", max_nfev=10000
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c, d = best_params
    return a, b, c, d


@file_cache()
def o1_simple_fit_power_law_two_terms_plus_e(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the two-term power law model with an added constant term using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = power_law_two_terms_plus_e
    else:
        y_transformed = y
        function = exp_power_law_two_terms_plus_e

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5, 1.0, 0.5, 0.1],
        [np.max(y_mean), 1.0, np.max(y_mean) / 2, 1.0, np.min(y_mean)],
        [0.1, 0.5, 0.1, 0.5, 0.0],
        [10.0, 1.0, 5.0, 0.5, 0.1],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals,
            initial_guess,
            args=(x, y_mean),
            bounds=(0, np.inf),
            loss="huber",
            max_nfev=20000,  # Increased maxfev for complex models
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c, d, e = best_params
    return a, b, c, d, e


@file_cache()
def o1_simple_fit_power_law_single_term_no_constant(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the single-term power law model without a constant term using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        x_log = np.log(x)
        y_log = np.log(y + epsilon)
        y_mean = y_log.mean(axis=0) if y_log.ndim > 1 else y_log

        def residuals(params, x_log, y):
            a_log, b = params
            return a_log - b * x_log - y

        initial_guesses = [
            [0.0, 0.0],
            [np.log(np.max(y_mean)), -1.0],
            [np.log(np.min(y_mean)), 1.0],
        ]

        best_loss = np.inf
        best_params = None

        for initial_guess in initial_guesses:
            result = least_squares(residuals, initial_guess, args=(x_log, y_mean), max_nfev=10000, loss="huber")
            if result.cost < best_loss:
                best_loss = result.cost
                best_params = result.x

        a_log, b = best_params
        a = np.exp(a_log)
    else:
        y_mean = y.mean(axis=0) if y.ndim > 1 else y

        def residuals(params, x, y):
            return power_law_single_term_no_constant(x, *params) - y

        initial_guesses = [
            [1.0, 0.5],
            [np.max(y_mean), 1.0],
            [0.1, 0.5],
        ]

        best_loss = np.inf
        best_params = None

        for initial_guess in initial_guesses:
            result = least_squares(
                residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), max_nfev=10000, loss="huber"
            )
            if result.cost < best_loss:
                best_loss = result.cost
                best_params = result.x

        a, b = best_params

    return a, b


@file_cache()
def o1_simple_fit_exp_decay_model(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the exponential decay model using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = exp_decay_model
    else:
        y_transformed = y
        function = exp_exp_decay_model

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5, 0.1],
        [np.max(y_mean), 0.5, np.min(y_mean)],
        [0.1, 0.1, 0.0],
        [10.0, 1.0, 0.1],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), max_nfev=10000, loss="huber"
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c = best_params
    return a, b, c


@file_cache()
def o1_simple_fit_exp_poly_model(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the exponential plus polynomial model using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = exp_poly_model
    else:
        y_transformed = y
        function = exp_exp_poly_model

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5, 1.0, 0.5, 0.1],
        [np.max(y_mean), 0.5, np.max(y_mean) / 2, 1.0, np.min(y_mean)],
        [0.1, 0.1, 0.1, 0.1, 0.0],
        [10.0, 1.0, 5.0, 0.5, 0.1],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals,
            initial_guess,
            args=(x, y_mean),
            bounds=(0, np.inf),
            max_nfev=20000,  # Increased maxfev for complex models
            loss="huber",
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c, d, e = best_params
    return a, b, c, d, e


@file_cache()
def o1_simple_fit_functional_form(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the custom functional form model using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = functional_form
    else:
        y_transformed = y
        function = exp_functional_form

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 1.0, 1.0, 0.1],
        [np.max(y_mean), 1.0, 1.0, np.min(y_mean)],
        [0.1, 0.1, 0.1, 0.0],
        [10.0, 5.0, 2.0, 0.1],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), max_nfev=10000, loss="huber"
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b, c, d = best_params
    return a, b, c, d


def power_law_no_constant(x, a, b):
    return a * np.power(x, -b)


def exp_power_law_no_constant(x, a, b):
    return np.exp(-power_law_no_constant(x, a, b))


@file_cache()
def o1_simple_fit_power_law_no_constant(x, y, fit_raw_data=True, epsilon=1e-5):
    """
    Fits the power law model without a constant term using robust optimization and multiple initial guesses.
    """
    if not fit_raw_data:
        y_transformed = -np.log(y + epsilon)
        function = power_law_no_constant
    else:
        y_transformed = y
        function = exp_power_law_no_constant

    y_mean = y_transformed.mean(axis=0) if y_transformed.ndim > 1 else y_transformed

    best_loss = np.inf
    best_params = None

    def residuals(params, x, y):
        return function(x, *params) - y

    initial_guesses = [
        [1.0, 0.5],
        [np.max(y_mean), 1.0],
        [0.1, 0.1],
        [10.0, 2.0],
    ]

    for initial_guess in initial_guesses:
        result = least_squares(
            residuals, initial_guess, args=(x, y_mean), bounds=(0, np.inf), loss="huber", max_nfev=10000
        )
        if result.cost < best_loss:
            best_loss = result.cost
            best_params = result.x

    a, b = best_params
    return a, b
