import functools
import tensorflow


def _allow_decorator_call_without_parentheses(function):
    """A decorator decorator, allowing to use the decorator to be used without
    parentheses if not arguments are provided. All arguments must be optional.
    """

    @functools.wraps(function)
    def decorator(*args, **kwargs):
        if len(args) == 1 and len(kwargs) == 0 and callable(args[0]):
            return function(args[0])
        else:
            return lambda wrapee: function(wrapee, *args, **kwargs)
    return decorator


@_allow_decorator_call_without_parentheses
class _CachedProperty(object):
    """A decorator that converts a function into a lazy property.  The
    function wrapped is called the first time to retrieve the result
    and then that calculated result is used the next time you access
    the value
    """

    # implementation detail: this property is implemented as non-data
    # descriptor.  non-data descriptors are only invoked if there is
    # no entry with the same name in the instance's __dict__.
    # this allows us to completely get rid of the access function call
    # overhead.  If one choses to invoke __get__ by hand the property
    # will still work as expected because the lookup logic is replicated
    # in __get__ for manual invocation.

    _MISSING = object()

    def __init__(self, func, always_recreated=False, name=None, doc=None):
        self.__name__ = name or func.__name__
        self.__module__ = func.__module__
        self.__doc__ = doc or func.__doc__
        self.func = func
        self.always_recreated = always_recreated
        #if not func.container:
        #    func.container = []
        #func.container.

    def __get__(self, obj, type=None):
        if obj is None:
            return self
        value = obj.__dict__.get(self.__name__, self._MISSING)
        if value is self._MISSING:
            value = self.func(obj)
            if self.always_recreated:
                return value
            obj.__dict__[self.__name__] = value
        return value


@_allow_decorator_call_without_parentheses
def property_scoped(f, scope=None, always_recreated=False, *args, **kwargs):
    """ The operations added by the function live within a tf.variable_scope(). If
    this decorator is used with arguments, they will be forwarded to the
    variable scope. The scope name defaults to the name of the wrapped
    function.
    """
    name = scope or f.__name__

    @_CachedProperty(always_recreated)
    @functools.wraps(f)
    def decorator(self):
        with tensorflow.variable_scope(name, *args, **kwargs):
            return f(self)

    return decorator
