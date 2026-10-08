import os
import sys
import subprocess
import ssl

# 1. Universal programmatic bypass for background threads crashing on the Windows Cert Store
def empty_load_certs(*args, **kwargs):
    pass
ssl.SSLContext.load_default_certs = empty_load_certs

def main():
    # 2. Check and enforce certifi bundle tracking
    try:
        import certifi
    except Exception:
        print("Missing dependency: certifi. Install with `pip install certifi`.", file=sys.stderr)
        sys.exit(1)

    os.environ['SSL_CERT_FILE'] = certifi.where()
    os.environ['PYTHONHTTPSVERIFY'] = "0"  # Enforce variable bypass across subprocess boundaries
    print("Using SSL_CERT_FILE =", os.environ['SSL_CERT_FILE'])

    # 3. Locate and execute app.py dynamically within the current environment
    try:
        # Determine script folder context safely [[1](https://docs.python.org/3/library/sys.html)]
        current_dir = os.path.dirname(os.path.abspath(__file__)) if '__file__' in locals() else os.getcwd()
        script = os.path.join(current_dir, 'app.py')
        
        args = [sys.executable, '-m', 'streamlit', 'run', script] + sys.argv[1:]
        subprocess.check_call(args)
    except subprocess.CalledProcessError as e:
        print(f"Streamlit exited with: {e}", file=sys.stderr)
        sys.exit(e.returncode)

if __name__ == '__main__':
    main()