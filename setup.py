
import setuptools

# force platform / python version specific wheel
class BinaryDistribution(setuptools.dist.Distribution):
    def has_ext_modules(self):
        return True

setuptools.setup(
    name="tick",
    version='0.7.0.2',
    author="Emmanuel Bacry, "
           "Stephane Gaiffas, "
           "Martin Bompaire, "
           "Søren V. Poulsen, "
           "Maryan Morel, "
           "Simon Bussy, "
           "Philip Deegan",
    author_email='martin.bompaire@polytechnique.edu, '
                 'philip.deegan@polytechnique.edu',
    url="https://x-datainitiative.github.io/tick/",
    description="Module for statistical learning, with a particular emphasis "
                "on time-dependent modelling",
    install_requires=['numpy',
                      'scipy',
                      'numpydoc',
                      'matplotlib',
                      'sphinx',
                      'pandas',
                      'dill',
                      'scikit-learn'],
    classifiers=['Development Status :: 3 - Alpha',
                 'Intended Audience :: Science/Research',
                 'Intended Audience :: Developers',
                 'Programming Language :: C++',
                 'Programming Language :: Python',
                 'Topic :: Software Development',
                 'Topic :: Scientific/Engineering',
                 'Operating System :: POSIX',
                 'Operating System :: Unix',
                 'Operating System :: MacOS',
                 'Programming Language :: Python :: 3.7',
                 'Programming Language :: Python :: 3.8',
                 'License :: OSI Approved :: BSD License'],
    include_package_data=True,
    packages=setuptools.find_packages(exclude=["lib/","tests/"]),
    distclass=BinaryDistribution,
)
