# src/system_control.py

class SystemControl:

    def __init__(self):

        self.locked = False

    def toggle_lock(self, gesture):

        # Lock

        if gesture == "lock" and not self.locked:

            self.locked = True

            print("[SYSTEM] Locked")

        # Unlock

        elif gesture == "unlock" and self.locked:

            self.locked = False

            print("[SYSTEM] Unlocked. Gestures active.")

    def is_locked(self):

        return self.locked
