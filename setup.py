"""A setuptools based setup module.

See:
https://packaging.python.org/en/latest/distributing.html
https://github.com/pypa/sampleproject
"""

import os
# To use a consistent encoding
from codecs import open

# Always prefer setuptools over distutils
from setuptools import setup, find_packages

here = os.path.abspath(os.path.dirname(__file__))

# Get the long description from the README file
with open(os.path.join(here, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# List run-time dependencies here.  These will be installed by pip when
# your project is installed. For an analysis of "install_requires" vs pip's
# requirements files see:
# https://packaging.python.org/en/latest/requirements.html
reqs = []
deps = []
with open(os.path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        if 'https://' not in line:
            reqs.append(line.strip())
        else:
            deps.append(line.strip())

reqs = list(filter(lambda elem: len(elem) > 0, reqs))
deps = list(filter(lambda elem: len(elem) > 0, deps))

setup(
    name='fast-abs-rl',

    # Versions should comply with PEP440.  For a discussion on single-sourcing
    # the version across setup.py and the project code, see
    # https://packaging.python.org/en/latest/single_source_version.html
    version='0.0.3',

    long_description=long_description,

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=[
        'Development Status :: 5 - Production/Stable',

        # Indicate who your project is intended for
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3.6',
    ],

    # What does your project relate to?
    keywords='abstractive summarization',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=find_packages(),

    install_requires=reqs,
    dependency_links=deps,

    # Include resource files
    include_package_data=True,

    entry_points={
        'console_scripts': [],
    },
)
