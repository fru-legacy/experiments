import tensorflow as tf
from cached_property import cached_property as cached_property_decorator


def scope(name=None, cached_property=False, *decorator_args, **decorator_kwargs):

    def instantiate(f):
        def f_scoped(self, *call_args, **call_kwargs):
            scope_name = name or f.__name__.replace('_', '')
            with tf.variable_scope(scope_name, *decorator_args, **decorator_kwargs):
                return f(self, *call_args, **call_kwargs)
        f_scoped.__name__ = f.__name__

        return cached_property_decorator(f_scoped) if cached_property else f_scoped

    # Make scope usable without parentheses
    if callable(name):
        func = name
        name = None
        return instantiate(func)

    return instantiate
