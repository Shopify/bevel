import os

from setuptools import setup


def filepath(fname):
    return os.path.join(os.path.dirname(__file__), fname)

exec(compile(open('eunomia/version.py').read(),
                  'eunomia/version.py', 'exec'))

readme_md = filepath('README.md')

try:
    import pypandoc
    readme_rst = pypandoc.convert_file(readme_md, 'rst')
except(ImportError):
    readme_rst = open(readme_md).read()


setup(
    name="eunomia",
    version=__version__,
    author="Ross Diener, Steven Wu, Cameron Davidson-Pilon",
    author_email="ross.diener@shopify.com ",
    description="Ordinal regression in Python",
    license="MIT",
    keywords="oridinal regression statistics data analysis",
    url="https://github.com/ShopifyHR/eunomia",
    packages=['eunomia',
              ],
    long_description=readme_rst,
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Topic :: Scientific/Engineering",
        ],
    install_requires=[
    ],
    package_data={
        "eunomia": [
            "../README.md",
            "../LICENSE",
        ]
    },
)
