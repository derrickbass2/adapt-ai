from setuptools import find_namespace_packages, setup

setup(
    name="user_service",
    version="0.1.0",
    packages=find_namespace_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        'Flask',
        'SQLAlchemy',
        'psycopg2-binary',
        'gunicorn'
    ],
    python_requires=">=3.8",
)