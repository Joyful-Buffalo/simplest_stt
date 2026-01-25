class Tee:
    def __init__(self,*fs):
        self.fs = list(fs)

    def write(self,s:str):
        for f in self.fs:
            try:
                if hasattr(f,"closed") and f.closed:
                    continue
                f.write(s)
                f.flush()
            except Exception:
                pass

    def flush(self):
        for f in self.fs:
            try:
                if hasattr(f,'closed') and f.closed:
                    continue
                f.flush()
            except Exception:
                pass
        