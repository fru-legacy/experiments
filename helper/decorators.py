import functools
import tensorflow


def _allow_decorator_call_without_parentheses(f):
    """A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(f)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return f(args[0])
        else:
            return lambda wrapee: f(wrapee, *args, **kwargs)

    return decorator


@_allow_decorator_call_without_parentheses
def variable_scope(f, scope=None, *args, **kwargs):
    """ Wrap the functions body in a tf.variable_scope(). If this decorator
    is used with arguments, they will be forwarded to the variable scope.
    The scope name defaults to the name of the wrapped function.
    """

    name = scope or f.__name__.replace('_', '')

    @functools.wraps(f)
    def decorator(self, *fargs, **fkwargs):
        with tensorflow.variable_scope(name, *args, **kwargs):
            return f(self, *fargs, **fkwargs)

    return decorator
