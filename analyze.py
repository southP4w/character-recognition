import time
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt


def compare_accuracies(X_test, y_test, *models):
    model_names = [model.__class__.__name__ for model in models]
    accuracies  = [model.score(X_test, y_test) for model in models]
    plt.figure()
    plt.bar(model_names, accuracies)
    plt.title('Model Accuracies')
    plt.ylabel('Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(accuracies):
        plt.text(i, v + .00000001, f'{v:.8f}', ha='center')
    plt.show()


def compare_average_accuracies(X_test, y_test, number_of_runs, *models):
    model_names = [model.__class__.__name__ for model in models]
    avg_scores  = []
    for model in models:
        scores  = [model.score(X_test, y_test) for _ in range(number_of_runs)]
        avg_scores.append(np.mean(scores))
    plt.figure()
    plt.bar(model_names, avg_scores)
    plt.title(f'Average Model Accuracies over {number_of_runs} runs')
    plt.ylabel('Average Accuracy')
    plt.ylim(0, 1)
    for i, v in enumerate(avg_scores):
        plt.text(i, v + .00000001, f'{v:.8f}', ha='center')
    plt.show()


def compare_runtime_and_mem(X_train, y_train, X_test, *models):
    model_names = [model.__class__.__name__ for model in models]
    runtimes    = []
    memories    = []
    for model in models:
        tracemalloc.start()
        start   = time.perf_counter()
        model.fit(X_train, y_train)
        model.predict(X_test)
        runtimes.append(time.perf_counter() - start)
        _, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        memories.append(peak / 1024**2)
    fig, (ax_rt, ax_mem)    = plt.subplots(1, 2, figsize=(12, 5))
    ax_rt.bar(model_names, runtimes)
    ax_rt.set_title('Model Runtime (s)')
    ax_rt.set_ylabel('Seconds')
    ax_rt.tick_params(axis='x', rotation=45, which='major', labelsize=10)
    for i, v in enumerate(runtimes):
        ax_rt.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    ax_mem.bar(model_names, memories)
    ax_mem.set_title('Peak Memory Usage (MB)')
    ax_mem.set_ylabel('MB')
    ax_mem.tick_params(axis='x', rotation=45, which='major', labelsize=10)
    for i, v in enumerate(memories):
        ax_mem.text(i, v, f'{v:.3f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.show()


def compare_to_sets(X, y, vectorized_image, *digit_sets):
    if not digit_sets:
        data    = X
    else:
        data    = np.vstack([X[y==digit_set] for digit_set in digit_sets])
    mean    = data.mean(axis=0)
    median  = np.median(data, axis=0)
    mode    = np.zeros(data.shape[1])
    for i in range(data.shape[1]):
        values, counts  = np.unique(data[:, i], return_counts=True)
        mode[i]         = values[counts.argmax()]
    similarity  = lambda v: 1 - np.mean((vectorized_image - v)**2)/4

    return similarity(mean), similarity(median), similarity(mode)


def predict_digit(classifier, handwritten_digit):
    return classifier.predict(handwritten_digit.reshape(1, -1))[0]
