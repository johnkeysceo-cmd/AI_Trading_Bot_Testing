class EventBus:
    """
    Simple synchronous event bus.
    """
    def __init__(self):
        self.listeners = []

    def register(self, callback):
        self.listeners.append(callback)

    def emit(self, event):
        for callback in self.listeners:
            callback(event)
