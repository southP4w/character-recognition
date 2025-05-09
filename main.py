import preprocess, analyze, classify, generate
from analyze import compare_to_sets

if __name__=='__main__':
    open('results.txt', 'w').close()
    training_pixels,    training_labels = preprocess.vectorize_dataset('dataset/train-data.txt')
    test_pixels,        test_labels     = preprocess.vectorize_dataset('dataset/test-data.txt')
    results                             = open('results.txt', 'w')

    """
    Sample and validation/misc functions
    """
    # generate.plot_random_samples(training_pixels, training_labels, 10)
    # generate.plot_mean_image(training_pixels, training_labels)
    # generate.plot_mean_image(training_pixels, training_labels, 9)
    # generate.plot_mean_image(training_pixels, training_labels, 2, 3, 5, 8)
    # generate.plot_median_image(training_pixels, training_labels)
    # generate.plot_median_image(training_pixels, training_labels, 9)
    # generate.plot_median_image(training_pixels, training_labels, 2, 3, 5, 8)
    # generate.plot_mode_image(training_pixels, training_labels)
    # generate.plot_mode_image(training_pixels, training_labels, 9)
    # generate.plot_mode_image(training_pixels, training_labels, 2, 3, 5, 8)

    """
    Try classifying your own image below
    """
    # normalized_image_vector = preprocess.image_to_grayscale_vector(
    #         path_to_image='your-characters/9/9_0.png',
    #         display_processed_image=True
    # )
    #
    # mean_comparison, median_comparison, mode_comparison = compare_to_sets(
    #         training_pixels,
    #         training_labels,
    #         normalized_image_vector,
    #         9
    # )
    # print(
    #         f'\nSimilarity of your handwritten digit to the mean (of the given set/sets) handwritten digit image:\n{mean_comparison}',
    #         f'\nSimilarity of your handwritten digit to the median (of the given set/sets) handwritten digit image:\n{median_comparison}',
    #         f'\nSimilarity of your handwritten digit to the mode (of the given set/sets) handwritten digit image:\n{mode_comparison}',
    #         '\n'
    # )

    """
    Non-Randomized Classifiers
    """
    # model_gnb   = classify.gaussian_naive_bayes_scikit(
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'GNB Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_gnb,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_knn   = classify.k_nearest_neighbors_scikit(
    #         k=3,
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'KNN Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_knn,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_lrc   = classify.logistic_regression_scikit(
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'LogReg Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_lrc,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_svc   = classify.support_vector_machine_scikit(
    #         C=10.579996,
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'SVM Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_svc,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_my_svc    = classify.support_vector_machine(
    #         10.579996,
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'(My) SVM Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_my_svc,
    #         handwritten_digit=normalized_image_vector
    # )}"')

    """
    Randomized Seed Classifiers
    """
    # model_dtc   = classify.decision_tree_scikit(
    #         max_depth=None,
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'DT Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_dtc,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_rdf   = classify.random_decision_forest_scikit(
    #         n=250,
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'RDF Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_rdf,
    #         handwritten_digit=normalized_image_vector
    # )}"')
    # model_mlp   = classify.multi_level_perceptron_scikit(
    #         X_train=training_pixels, y_train=training_labels,
    #         X_test=test_pixels, y_test=test_labels,
    #         display_conf_matx=True
    # )
    # print(f'MLP Classifier: "You gave me a handwritten {analyze.predict_digit(
    #         classifier=model_mlp,
    #         handwritten_digit=normalized_image_vector
    # )}"')

    """
    Plot graphs comparing accuracies and performance
    """
    # analyze.compare_accuracies(
    #         test_pixels,
    #         test_labels,
    #         model_gnb,
    #         model_knn,
    #         model_lrc,
    #         model_svc,
    #         model_my_svc,
    #         model_dtc,
    #         model_rdf,
    #         model_mlp
    # )
    # analyze.compare_average_accuracies(
    #         test_pixels,
    #         test_labels,
    #         100,
    #         model_dtc,
    #         model_rdf,
    #         model_mlp
    # )
    # analyze.compare_runtime_and_mem(
    #         training_pixels, training_labels,
    #         test_pixels,
    #         model_gnb,
    #         model_knn,
    #         model_lrc,
    #         model_svc,
    #         model_my_svc,
    #         model_dtc,
    #         model_rdf,
    #         model_mlp
    # )

    results.close()
