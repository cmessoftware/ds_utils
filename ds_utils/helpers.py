import sys, os

def running_context() -> str:
    # Orden de chequeo
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell:
            name = shell.__class__.__name__
            if name == "ZMQInteractiveShell":
                # Señales de Colab/Kaggle
                if "COLAB_GPU" in os.environ:
                    return "colab_notebook"
                if "KAGGLE_KERNEL_RUN_TYPE" in os.environ:
                    return "kaggle_notebook"
                return "jupyter_notebook"
            if name == "TerminalInteractiveShell":
                return "ipython_terminal"
    except Exception:
        pass

    # Señales de lanzamiento vía ipykernel
    if "ipykernel" in sys.modules or any("ipykernel_launcher" in a for a in sys.argv):
        return "jupyter_notebook"

    # Si nada matchea, asumimos script
    return "python_script"
