#Space for imports

class Player:
    def __init__(self, name):
        self.name = name
        self.id = 0 #ID assigned by deepsort
        self.bbox = [] #boundary Box (Format: [lt-x, lt-y, rb-x, rb-y])
        self.missing_frames = 0 #consecutive frames this player hasn't been seen, so update_players() can free their slot
