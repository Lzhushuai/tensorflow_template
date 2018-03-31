import logging
import doctest
import functools
import tensorflow as tf


def set_logging_basic_config(**kwargs):
    if 'format' not in kwargs:
        kwargs['format'] = '[%(name)s] : %(asctime)s : %(levelname)s : %(message)s'
    if 'level' not in kwargs:
        kwargs['level'] = logging.INFO

    logging.basicConfig(**kwargs)


def with_graph(graph):
    """

    Examples:
        >>> g = tf.Graph()

        >>> @with_graph(g)
        ... def model_part1():
        ...     a = tf.Variable(1)
        ...     return a

        >>> @with_graph(g)
        ... def model_part2():
        ...     b = tf.Variable(2)
        ...     return b

        >>> a = model_part1()
        >>> b = model_part2()
        >>> with g.as_default():
        ...     c = a + b

        >>> with tf.Session(graph=g) as sess:
        ...     tf.global_variables_initializer().run()
        ...     ret = sess.run(c)
        ...     print(ret)
        3

        >>> print(model_part1.__name__)
        model_part1

    Args:
        graph(tf.Graph):

    Returns:

    """
    def _with_graph(func):

        @functools.wraps(func)
        def func_with_graph(*args, **kwargs):

            with graph.as_default():
                ret = func(*args, **kwargs)

            return ret

        return func_with_graph

    return _with_graph


if __name__ == '__main__':
    """"""
    doctest.testmod()