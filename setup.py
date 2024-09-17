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
        'gunicorn',
        'pyspark==3.5.2',  # Fixed syntax error for specifying the pyspark version
        'pydantic',
        'python-dotenv',
        'Flask-JWT-Extended',
        'Flask-Migrate',
        'tensorflow',
        'authlib',
        'Werkzeug',
        'pytest',
        'pandas',
        'alembic',
    ],
    python_requires=">=3.8",
)
