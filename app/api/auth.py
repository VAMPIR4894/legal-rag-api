import os
from flask_httpauth import HTTPBasicAuth
from werkzeug.security import check_password_hash

# Initialize Flask-HTTPAuth
auth = HTTPBasicAuth()

# Get credentials from environment variables (using defaults for testing)
USER = os.environ.get("BASIC_AUTH_USER", "legal_user")
PASS = os.environ.get("BASIC_AUTH_PASS", "super_secure_password123")

# Note: We use a simple hash comparison here. 
# In a real app, the password should be stored as a strong hash (e.g., bcrypt)
# and compared using check_password_hash.

# Using a fake hash for demonstration, assuming the environment provides the cleartext password for comparison
# In production, the password would be hashed *before* being stored/checked.
# For simplicity, we compare cleartext passwords from the environment.
@auth.verify_password
def verify_password(username, password):
    """
    Verifies the provided username and password against environment variables.
    """
    if username == USER and password == PASS:
        return username
    return None