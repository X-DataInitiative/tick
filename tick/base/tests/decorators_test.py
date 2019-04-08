# License: BSD 3 clause

from tick.base import actual_kwargs

import unittest


class Test(unittest.TestCase):
    def test_actual_kwargs(self):
        """...Test actual_kwargs decorator
        """

        @actual_kwargs
        def f(arg1, arg2, kwarg1=None, kwarg2='', kwarg3=1, kwarg4=True):
            kwargs_ = sorted(f.actual_kwargs.items())
            all_kwargs_ = sorted({
                'arg1': arg1,
                'arg2': arg2,
                'kwarg1': kwarg1,
                'kwarg2': kwarg2,
                'kwarg3': kwarg3,
                'kwarg4': kwarg4
            }.items())
            return kwargs_, all_kwargs_

        arg1 = 1
        arg2 = 2
        default_all_kwargs = [('arg1', arg1), ('arg2', arg2), ('kwarg1', None),
                              ('kwarg2', ''), ('kwarg3', 1), ('kwarg4', True)]

        kwargs, all_kwargs = f(arg1, arg2)
        self.assertEqual(kwargs, [])
        self.assertEqual(all_kwargs, default_all_kwargs)

        kwargs, all_kwargs = f(arg1, arg2, kwarg1='value')
        self.assertEqual(kwargs, [('kwarg1', 'value')])
        expect_all_kwargs = default_all_kwargs.copy()
        expect_all_kwargs[2] = ('kwarg1', 'value')
        self.assertEqual(all_kwargs, expect_all_kwargs)

        kwargs, all_kwargs = f(arg1, arg2, kwarg2='value2', kwarg3=-3)
        self.assertEqual(kwargs, [('kwarg2', 'value2'), ('kwarg3', -3)])
        expect_all_kwargs = default_all_kwargs.copy()
        expect_all_kwargs[3:5] = [('kwarg2', 'value2'), ('kwarg3', -3)]
        self.assertEqual(all_kwargs, expect_all_kwargs)

        kwargs, all_kwargs = f(-2, arg2, kwarg4=False)
        self.assertEqual(kwargs, [('kwarg4', False)])
        expect_all_kwargs = default_all_kwargs.copy()
        expect_all_kwargs[0] = ('arg1', -2)
        expect_all_kwargs[5] = ('kwarg4', False)
        self.assertEqual(all_kwargs, expect_all_kwargs)

        msg = "^f\(\) got an unexpected keyword argument 'kwarg5'$"
        with self.assertRaisesRegex(TypeError, msg):
            f(arg1, arg2, kwarg5='un_existing')


if __name__ == "__main__":
    unittest.main()
