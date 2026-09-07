"""Credit default risk model: data prep, features, training and evaluation.

``src.features`` is imported at unpickle time by the serialized model, so this
package must stay importable in the serving container.  That is why nothing
here imports Optuna, SHAP or matplotlib at module level - those live inside
``src.train``, which the API never imports.
"""
