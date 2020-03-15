from hyperopt import hp, fmin, tpe


def objective(x):
    print(x)
    return x ** 2


def get_space():
    return {
        'c1': hp.uniform('c1', -5e-5, 5e-5),
        'c2': hp.uniform('c2', -1e-2, 1e-2),
        'c3': hp.uniform('c3', -1000, 1000),
    }


def main():
    best = fmin(objective,
                space=get_space(),
                algo=tpe.suggest,
                max_evals=100)
    print(best)


if __name__ == '__main__':
    main()
