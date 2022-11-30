from setuptools import setup, find_packages

VERSION = '0.0.1' 
DESCRIPTION = 'Easy-ICD'
LONG_DESCRIPTION = 'Easily create image classification datasets'

# Setting up
setup(
       # the name must match the folder name 'verysimplemodule'
        name="easy_icd", 
        version=VERSION,
        author="Sashwat Anagolum",
        author_email="sashwat.anagolum@gmail.com",
        description=DESCRIPTION,
        long_description=LONG_DESCRIPTION,
        packages=find_packages(),
        install_requires=[], # add any additional packages that 
        # needs to be installed along with your package. Eg: 'caer'
        
        keywords=['python'],
        classifiers= [
            "Development Status :: 3 - Alpha",
            "Programming Language :: Python :: 3",
            "Operating System :: MacOS :: MacOS X",
            "Operating System :: Microsoft :: Windows",
        ]
)