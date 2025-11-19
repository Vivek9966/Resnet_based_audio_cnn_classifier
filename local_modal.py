# local_modal.py
class App:
    def __init__(self, name):
        self.name = name
        print(f"[LOCAL MODAL] App initialized: {name}")

    def function(self, **kwargs):
        def decorator(fn):
            print(f"[LOCAL MODAL] Registered function: {fn.__name__}")
            return fn
        return decorator

    def cls(self, **kwargs):
        def decorator(cls):
            print(f"[LOCAL MODAL] Registered class: {cls.__name__}")
            return cls
        return decorator
