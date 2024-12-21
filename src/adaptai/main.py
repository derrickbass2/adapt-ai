import os

from adaptai.routes import app


def main():
    """
    Entry point for the Flask application.
    - Configures the host, port, and debug settings from environment variables.
    - Starts the Flask app.
    """
    # Get configuration from environment variables with default fallbacks
    host = os.getenv("FLASK_HOST", "0.0.0.0")  # Default host allows access from any network
    port = int(os.getenv("FLASK_PORT", 5000))  # Default port is 5000
    debug = bool(int(os.getenv("FLASK_DEBUG", 1)))  # 1 = debug enabled, 0 = debug disabled

    # Start the Flask app
    app.run(host=host, port=port, debug=debug)


if __name__ == "__main__":
    main()
