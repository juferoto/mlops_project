from sklearn.pipeline import Pipeline

from preprocessors import CustomLogisticRegression, ImageTransformation


def create_pipeline(input_dir, categories):
    image_normalizer = ImageTransformation(input_dir, categories)

    # Llama a remove_background antes del metodo image_normalize()
    # image_normalizer.remove_background()

    pipeline = Pipeline(
        [("normalize", image_normalizer), ("model", CustomLogisticRegression())]
    )

    return pipeline
